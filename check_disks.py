#!/usr/bin/python
#
# Check all disks in a host
#
# @author:  Piotr Kasprzak, piotr.kasprzak@gwdg.de

import sys
import stat
import errno
import logging
import time
import uuid
import os
import os.path
import platform
import re
import subprocess
import atexit
import argparse
import pprint
import time

from multiprocessing.dummy import Pool as ThreadPool
from Queue import PriorityQueue

# Stuff from ceph-deploy
from ceph_deploy.lib import remoto

LOG = logging.getLogger(os.path.basename(sys.argv[0]))

###### exceptions ########

class Error(Exception):
    """
    Error
    """

    def __str__(self):
        doc = self.__doc__.strip()
        return ': '.join([doc] + [str(a) for a in self.args])

# ---------- Util stuff liberaly taken from ceph-disk

def which(executable):
    """find the location of an executable"""
    if 'PATH' in os.environ:
        envpath = os.environ['PATH']
    else:
        envpath = os.defpath
    PATH = envpath.split(os.pathsep)

    locations = PATH + [
        '/usr/local/bin',
        '/bin',
        '/usr/bin',
        '/usr/local/sbin',
        '/usr/sbin',
        '/sbin',
    ]

    for location in locations:
        executable_path = os.path.join(location, executable)
        if os.path.exists(executable_path):
            return executable_path


def _get_command_executable(arguments):
    """
    Return the full path for an executable, raise if the executable is not
    found. If the executable has already a full path do not perform any checks.
    """
    if arguments[0].startswith('/'):  # an absolute path
        return arguments
    executable = which(arguments[0])
    if not executable:
        command_msg = 'Could not run command: %s' % ' '.join(arguments)
        executable_msg = '%s not in path.' % arguments[0]
        raise ExecutableNotFound('%s %s' % (executable_msg, command_msg))

    # swap the old executable for the new one
    arguments[0] = executable
    return arguments


def run_command_and_block(arguments, args, **kwargs):

    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s' % ' '.join(arguments))
    if not args.dry_run:
        process = subprocess.Popen(
            arguments,
            stdout = subprocess.PIPE,
            **kwargs)
        stdout, stderr = process.communicate()
        return stdout, process.returncode
    else:
        # Return fake data
        return '', 0

def run_command_background(arguments, args, log_file, **kwargs):
    
    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s' % ' '.join(arguments))
    if not args.dry_run:
        process = subprocess.Popen(
            arguments,
            stdout = log_file,
            stderr = subprocess.STDOUT,
            **kwargs)
        return process
    else:
        return None

def command_check_call(arguments, args):

    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s', ' '.join(arguments))
    if not args.dry_run:
        return subprocess.check_call(arguments)
    else:
        return 0

def get_dev_size(dev, size='megabytes'):
    """
    Attempt to get the size of a device so that we can prevent errors
    from actions to devices that are smaller, and improve error reporting.

    :param size: bytes or megabytes
    :param dev: the device to calculate the size
    """
    fd = os.open(dev, os.O_RDONLY)
    dividers = {'bytes': 1, 'megabytes': 1024*1024}
    try:
        device_size = os.lseek(fd, 0, os.SEEK_END)
        divider = dividers.get(size, 1024*1024)  # default to megabytes
        return device_size/divider
    except Exception as error:
        LOG.warning('failed to get size of %s: %s' % (dev, str(error)))
    finally:
        os.close(fd)


def mkdir(*a, **kw):
    """
    Creates a new directory if it doesn't exist, removes
    existing symlink before creating the directory.
    """
    # remove any symlink, if it is there..
    if os.path.exists(*a) and stat.S_ISLNK(os.lstat(*a).st_mode):
        LOG.debug('Removing old symlink at %s', *a)
        os.unlink(*a)
    try:
        os.mkdir(*a, **kw)
    except OSError, e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def get_dev_name(path):
    """
    get device name from path.  e.g.::

        /dev/sda -> sdas, /dev/cciss/c0d1 -> cciss!c0d1

    a device "name" is something like::

        sdb
        cciss!c0d1

    """
    assert path.startswith('/dev/')
    base = path[5:]
    return base.replace('/', '!')

def remote_is_mounted(connection, device, args):
    """
    Check if the given device is mounted.
    """
    device = remote_path_realpath(connection, device, args)
    (stdout, exit_code) = remote_run_process_and_check(connection, ['cat', '/proc/mounts'], args)

    for line in stdout: 
        fields = line.split()
        if len(fields) < 3:
            continue
        mounts_dev = fields[0]
        path = fields[1]
        if mounts_dev.startswith('/') and remote_path_exists(connection, mounts_dev, args):
            mounts_dev = remote_path_realpath(connection, mounts_dev, args)
            if mounts_dev == device:
                return path

    return None

def remote_is_held(connection, device, args):
    """
    Check if a device is held by another device (e.g., a dm-crypt mapping)
    """
    assert remote_path_exists(connection, device, args)
    device = remote_path_realpath(connection, device, args)
    base = get_dev_name(device)

    # full disk?
    directory = '/sys/block/{base}/holders'.format(base=base)
    if remote_path_exists(connection, directory, args):
        return remote_listdir(connection, directory, args)

    # partition?
    part = base
    while len(base):
        directory = '/sys/block/{base}/{part}/holders'.format(part=part, base=base)
        if remote_path_exists(connection, directory, args):
            return remote_listdir(connection, directory, args)
        base = base[:-1]
    return []

def get_dev_path(name):
    """
    get a path (/dev/...) from a name (cciss!c0d1)
    a device "path" is something like::

        /dev/sdb
        /dev/cciss/c0d1

    """
    return '/dev/' + name.replace('!', '/')

def remote_list_partitions(connection, basename, args):
    """
    Return a list of partitions on the given device name
    """
    partitions = []
    for name in remote_listdir(connection, os.path.join('/sys/block', basename), args):
        if name.startswith(basename):
            partitions.append(name)
    return partitions

def remote_is_partition(connection, device, args):
    """
    Check whether a given device path is a partition or a full disk.
    """
    device = remote_path_realpath(connection, device, args)

#    if not stat.S_ISBLK(os.lstat(dev).st_mode):
#        raise Error('not a block device', dev)

    name = get_dev_name(device)
    if remote_path_exists(connection, '/sys/block/' + name, args):
        return False

    # make sure it is a partition of something else
    for basename in remote_listdir(connection, '/sys/block', args):
        if remote_path_exists(connection, '/sys/block/' + basename + '/' + name):
            return True

    raise Error('not a disk or partition', device)

def remote_device_not_in_use(connection, device, args, check_partitions = True):
    """
    Verify if a given device (path) is in use (e.g. mounted or
    in use by device-mapper).
    """
    assert remote_path_exists(connection, device, args)

    LOG.info('*** Checking if device "%s" is in use', device)

    if remote_is_mounted(connection, device, args):
        return False

    holders = remote_is_held(connection, device, args)

    if holders:
        return False

    if check_partitions and not remote_is_partition(connection, device, args):
        LOG.info('*** Checking partitions...')
        basename = get_dev_name(remote_path_realpath(connection, device, args))
        for partname in remote_list_partitions(connection, basename, args):
            partition = get_dev_path(partname)
            if remote_is_mounted(connection, partition, args):
                LOG.info('*** Partition "%s" mounted', partition)
                return False
            holders = remote_is_held(connection, partition, args)
            if holders:
                return False

    return True

# ----- Util stuff from ceph-deploy

def get_connection(hostname, username, logger, threads=5, use_sudo=None, detect_sudo=True):
    """
    A very simple helper, meant to return a connection
    that will know about the need to use sudo.
    """
    if username:
        hostname = "%s@%s" % (username, hostname)
    try:
        conn = remoto.Connection(
            hostname,
            logger=logger,
            threads=threads,
            detect_sudo=detect_sudo,
        )

        # Set a timeout value in seconds to disconnect and move on
        # if no data is sent back.
        conn.global_timeout = 300
        logger.debug("connected to host: %s " % hostname)
        return conn

    except Exception as error:
        msg = "connecting to host: %s " % hostname
        errors = "resulted in errors: %s %s" % (error.__class__.__name__, error)
        raise RuntimeError(msg + errors)

def _remote_get_command_executable(connection, arguments):
    """
    Return the full path for an executable, raise if the executable is not
    found. If the executable has already a full path do not perform any checks.
    """

    if arguments[0].startswith('/'):  # an absolute path
        return arguments

    # Run 'which' remotely to get the full path
    (stdout, stderr, exit_code) = remoto.process.check(connection, ['which', arguments[0]])

    executable = stdout[0]
#    LOG.info('>>> output: %s, %s, %s', stdout, stderr, exit_code)
    if not executable:
        command_msg = 'Could not run command: %s' % ' '.join(arguments)
        executable_msg = '%s not in path.' % arguments[0]
        raise ExecutableNotFound('%s %s' % (executable_msg, command_msg))

    # swap the old executable for the new one
    arguments[0] = executable

    return arguments

def remote_run_process_and_check(connection, arguments, args, **kwargs):

    arguments = _remote_get_command_executable(connection, arguments)
    LOG.debug('Arguments: %s', pprint.pformat(arguments))
    LOG.info('Running command: %s' % ' '.join(arguments))
    if not args.dry_run:
        (stdout, stderr, exit_code) = remoto.process.check(connection, arguments, **kwargs)
        LOG.debug('stdout: %s', pprint.pformat(stdout))
        return stdout, exit_code
    else:
        # Return fake data
        return '', 0

# ---------- Own util stuff

def remote_path_realpath(connection, path, args):

    (stdout, exit_code) = remote_run_process_and_check(connection, ['readlink', '-f', '-n', path], args)

    return stdout[0]

def remote_path_exists(connection, path, args):

    (stdout, exit_code) = remote_run_process_and_check(connection, ['readlink', '-f', '-n', path], args)

    if exit_code == 0:
        return True
    
    return False

def remote_listdir(connection, path, args):

    (stdout, exit_code) = remote_run_process_and_check(connection, ['ls', path], args)

#    dir_entries = str.split(stdout, ', ')

    return stdout


def device_is_ssd(connection, device, args):
    """
    Check if given device is an ssd (device needs to be referenced as /dev/*)
    """

    device = remote_path_realpath(connection, device, args)
#    if not stat.S_ISBLK(os.lstat(dev).st_mode):
#        raise Error('not a block device', dev)

    device_name = get_dev_name(device)
    sys_path    = '/sys/block/' + device_name + '/queue/rotational'

    if not remote_path_exists(connection, sys_path, args):
       raise Error('Could not check if device "%s" is a ssd: path "%s" does not exist!', device, sys_path)

    # Get info from sysfs
    (rotational_raw, exit_code) = remote_run_process_and_check(connection, ['cat', sys_path], args)
#    rotational_raw = open(sys_path).read().rstrip('\n')

    LOG.debug('Rotational status for device "%s": "%s"', device, rotational_raw[0])
    if rotational_raw[0] == '1':
        return False
    else:
        return True

def device_is_rotational(connection, device, args):
    return not device_is_ssd(connection, device, args)

processes = []

@atexit.register
def kill_subprocesses():
    for process in processes:
        process.kill()

# List of selected devices referencing devices in /dev directly
devices         = []

# List of selected devices referencing devices by id (i.e. from /dev/disk/by-id)
devices_by_id   = []

DEVICES = '^(ata-HDS)|(ata-ST3000)'

LOG_DIR = '/tmp'

THREADS = 30

BADBLOCKS_CALL = [ 'badblocks', '-b', '4096', '-v', '-w']

def select_devices(connection, args, remove_in_use = True):

    # List of selected devices referencing devices in /dev directly
    devices         = []

    # List of selected devices referencing devices by id (i.e. from /dev/disk/by-id)
    devices_by_id   = []

    LOG.info('Building device selection...')

    # Process selection by-id
    if args.disks_by_id:
        LOG.info('Adding device selection by id...')
        for device in remote_listdir(connection, '/dev/disk/by-id', args):

            # Ignore partitions
            if re.match(r'.*-part[0-9]+$', device):
                LOG.debug('Ignoring partition: %s', device)
                continue

            # Match devices
            if re.match(r'%s' % args.disks_by_id, device):
                LOG.info('Adding device to list: %s', device)
                devices.append(remote_path_realpath(connection, '/dev/disk/by-id/' + device, args))

    # Process selection by-path
    if args.disks_by_path:
        LOG.info('Adding device selection by path...')
        for device in remote_listdir(connection, '/dev/disk/by-path', args):
            
            # Ignore partitions
            if re.match(r'.*-part[0-9]+$', device):
                LOG.debug('Ignoring partition: %s', device)
                continue

            # Match devices
            if re.match(r'%s' % args.disks_by_path, device):
                LOG.info('Adding device to list: %s', device)
                devices.append(remote_path_realpath(connection, '/dev/disk/by-path/' + device, args))

    # Process selection by-partlabel
    if args.disks_by_partlabel:
        LOG.info('Adding device selection by partlabel...')
        for device in remote_listdir(connection, '/dev/disk/by-partlabel', args):
            
            # Match devices
            if re.match(r'%s' % args.disks_by_partlabel, device):
                LOG.info('Adding device to list: %s', device)
                devices.append(remote_path_realpath(connection, '/dev/disk/by-partlabel/' + device, args))

    # Remove dupes
    devices = list(set(devices))

    # Translate to by-id devices for usage in commands

    # First build by-id -> /dev mapping from /dev/disk/by-id
    by_id_to_dev_map = {}
    for device in remote_listdir(connection, '/dev/disk/by-id', args):
            
        # Ignore partitions
        if re.match(r'.*-part[0-9]+$', device):
#            LOG.debug('Ignoring partition: %s', device)
            continue

        # Match devices
        if re.match(r'%s' % args.disks_by_id, device):
#            LOG.info('Adding device to list: %s', device)
            by_id_to_dev_map[device] = remote_path_realpath(connection, '/dev/disk/by-id/' + device, args)
            LOG.debug("Mapping device '%s' to '%s'", device, by_id_to_dev_map[device])

    # Create inverse map
#    pp = pprint.PrettyPrinter(indent=4)
    LOG.debug('by_id_to_dev_map: %s', pprint.pformat(by_id_to_dev_map))

    dev_to_by_id_map = {v: k for k, v in by_id_to_dev_map.items()}
    LOG.debug('dev_to_by_id_map: %s', pprint.pformat(dev_to_by_id_map))
 
    # Translate devices into by-id format (absolute path included)
    for device in devices:

        device_by_id = dev_to_by_id_map[device]

        if args.disks_only_rotational and not device_is_rotational(connection, device, args):
            LOG.debug('Ignoring device "%s" as it is not rotational and --disks-only-rotational=true', device_by_id)
            continue

        if args.disks_only_ssds and not device_is_ssd(connection, device, args):
            LOG.debug('Ignoring device "%s" as it is not ssd and --disks-only-ssds=true', device_by_id)
            continue

        if remove_in_use:
            if remote_device_not_in_use(connection, device, args):
                devices_by_id.append('/dev/disk/by-id/' + device_by_id)
            else:
                # Do not consider devices which are currently in use
                LOG.debug('Ignoring device "%s" as it is currently is use', device_by_id)

    return devices_by_id

def check_disk(device):

    device_path = DEVICES_PATH + '/' + device
    LOG.info('Running badblocks on device: %s', device_path)

#    if not stat.S_ISBLK(os.lstat(device).st_mode):
#        raise Error('not a block device', device)

    file_name = LOG_DIR + '/badblocks/' + device

    badblocks_file = file_name + '.badblocks'
    LOG.debug('Using file for badblocks data: %s', badblocks_file)

    log_file_name = file_name + '.log'
    log_file = open(log_file_name, 'w')
    LOG.debug('Using log file for badblocks command: %s', log_file_name)

    badblocks_call = BADBLOCKS_CALL[:]
    badblocks_call.extend(['-o', badblocks_file, device_path])

    process = run_command_background(badblocks_call, log_file)
    processes.append(process)

    out, _ = process.communicate()

    processes.remove(process)

# ------ Main

def main_debug(args):

    LOG.info('Running debug...')

    # Get the list of selected devices
    connection = get_connection(args.host, args.user, LOG)
    devices_by_id = select_devices(connection, args)

    # Dump selected disks
    LOG.info('Selected disks:')
    for device in devices_by_id:
        LOG.info('- %s', device)

def main_ceph_deploy_osd_prepare(args):

    LOG.info('Running ceph deploy osd prepare...')

    # Get the list of selected devices
    connection = get_connection(args.host, args.user, LOG)
    devices_by_id = select_devices(connection, args)

    # Get list of journal devices to use
    journal_devices = str.split(args.journal_devices, ',')
    LOG.info('Using "%s" as journal devices to create partitions on', pprint.pformat(journal_devices))

    # Create array to hold number of partitions for each device to select the least used journal device for the next osd
    journal_devices_partitions = [None] * len(journal_devices)
    for i in range(0, len(journal_devices)):
        device_base_name = get_dev_name(journal_devices[i])
        partitions = remote_list_partitions(connection, device_base_name, args)
        journal_devices_partitions[i] = len(partitions)
        LOG.debug('Current number of partitions on journal device "%s": %i', device_base_name, len(partitions))

    # For each selected disk determine the least used journal device and use that in 'ceph-deploy prepare disk' call
    for osd_device in devices_by_id:
        
        # Find least used journal device from list
        best = 0
        for i in range(0, len(journal_devices)):
            if journal_devices_partitions[i] < journal_devices_partitions[best]:
                best = i
        LOG.debug('Using journal device "%s" for osd device "%s"', journal_devices[best], osd_device)

        osd_location = args.host + ':' + osd_device

        # Zap osd device first
        ceph_deploy_call = [ 'ceph-deploy', '--overwrite-conf', 'disk', 'zap', osd_location ]

        process = command_check_call(ceph_deploy_call, args)

        # Run 'ceph-deploy osd prepare'
        ceph_deploy_call = [ 'ceph-deploy', '--overwrite-conf', 'osd', 'prepare', osd_location + ':' + journal_devices[best] ]

        process = command_check_call(ceph_deploy_call, args)

        journal_devices_partitions[best] += 1
        LOG.debug('Waiting 5 secs...')
        time.sleep(5)

def main_badblocks(args):

    LOG.info('Running badblocks...')

    # Create log dir for badblocks
    mkdir(LOG_DIR + '/badblocks')

    # Get the list of devices to run badblocks on
    devices_by_id = select_devices(args)

    pool = ThreadPool(THREADS)

    try:
        results = pool.map(check_disk, devices_by_id)

    except KeyboardInterrupt:
        kill_subprocesses()
        sys.exit()

    pool.close()
    pool.join()
    

def parse_args():
 
    # Global arguments

    parser = argparse.ArgumentParser(
        'check_disks.py', )

    parser.add_argument(
        '-v', '--verbose',
        action          = 'store_true', default = None,
        help            = 'be more verbose', )

    parser.add_argument(
        '--logdir',
        metavar         = 'PATH',
        default         = '/tmp',
        help            = 'write log / output files to this dir (default /tmp)', )

    parser.add_argument(
        '--dry-run',
        action          = 'store_true',
        default         = False,
        help            = 'Do not modify system state, just print commands to be run (default false)', )

    parser.add_argument(
        '--host',
        metavar         = 'HOST',
        default         = 'localhost',
#        required        = True,
        help            = 'Host to run commands on (needs keyless ssh setup)', )

    parser.add_argument(
        '--user',
        metavar         = 'USER',
        required        = True,
        help            = 'Host to run commands on (needs keyless ssh setup)', )

    parser.add_argument(
        '--disks-by-id',
        metavar         = 'DISKS',
#        default         = '',
        help            = 'Select disks by id via regexp against /dev/disk/by-id (default none)', )

    parser.add_argument(
        '--disks-by-path',
        metavar         = 'DISKS',
#        default         = '',
        help            = 'Select disks by path via regexp against /dev/disk/by-path (default none)', )

    parser.add_argument(
        '--disks-by-partlabel',
        metavar         = 'DISKS',
#        default        = '',
        help            = 'Select disks by partlabel via regexp against /dev/disk/by-partlabel (default none)', )

    parser.add_argument(
        '--disks-only-rotational',
#        metavar        = 'BOOLEAN',
        action          = 'store_true',
        default         = False,
        help            = 'Only select rotational disks by selectors (default false)', )

    parser.add_argument(
        '--disks-only-ssds',
#        metavar = 'BOOLEAN',
        action  = 'store_true',
        default = False,
        help    = 'Only select ssd disks by selectors (default false)', )

    parser.set_defaults(
        # we want to hold on to this, for later
        prog = parser.prog,
#        disks_only_ssds=True,
#        cluster='ceph', 
    )

    subparsers = parser.add_subparsers(
        title           = 'subcommands',
        description     = 'valid subcommands',
        help            = 'sub-command help', )

    # badblocks related arguments

    badblocks_parser = subparsers.add_parser('badblocks', help='Run badblocks for selection of disks')

#    badblocks_parser.add_argument(
#        '--cluster',
#        metavar='NAME',
#        help='cluster name to assign this disk to',)

    badblocks_parser.set_defaults(
        function=main_badblocks, )

    # debug related arguments

    debug_parser = subparsers.add_parser('debug', help='Run debug')

#    debug_parser.add_argument(
#        '--cluster',
#        metavar='NAME',
#        help='cluster name to assign this disk to',)

    debug_parser.set_defaults(
        function=main_debug, )

    # ceph-deploy related arguments

    ceph_deploy_osd_prepare_parser = subparsers.add_parser('ceph-deploy-osd-prepare', help='Run "ceph-deploy osd prepare" on selected disks')

#    ceph_deploy_osd_prepare_parser.add_argument(
#        '--host',
#        metavar         = 'HOST',
#        required        = True,
#        help            = 'Host to run ceph-deploy on', )

    ceph_deploy_osd_prepare_parser.add_argument( 
        '--journal-devices',
        metavar         = 'DEVICE1,DEVICE2, ...',
        required        = True,
        help            = 'List of block device to create journal partitions on (will be used round robin)', )

    ceph_deploy_osd_prepare_parser.set_defaults(
        function=main_ceph_deploy_osd_prepare, )

    args = parser.parse_args()

    # Dump args array
    LOG.info('Current args: %s', pprint.pformat(args))

    return args

def main():

    args = parse_args()

    loglevel = logging.INFO
    if args.verbose:
        loglevel = logging.DEBUG

    # Initialize logging

#    LOG = logging.getLogger(os.path.basename(sys.argv[0]))
    LOG.setLevel(loglevel)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)

    LOG.addHandler(ch)


#    if args.prepend_to_path != '':
#        path = os.environ.get('PATH', os.defpath)
#        os.environ['PATH'] = args.prepend_to_path + ":" + path

    try:
        args.function(args)

    except Error as e:
        raise SystemExit(
            '{prog}: {msg}'.format(
                prog=args.prog,
                msg=e,
            )
        )

#    except CephDiskException as error:
#        exc_name = error.__class__.__name__
#        raise SystemExit(
#            '{prog} {exc_name}: {msg}'.format(
#                prog=args.prog,
#                exc_name=exc_name,
#                msg=error,
#            )
#        )


if __name__ == '__main__':
    main()


