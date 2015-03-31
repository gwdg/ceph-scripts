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

from multiprocessing.dummy import Pool as ThreadPool
from Queue import PriorityQueue

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


def run_command_and_block(arguments, **kwargs):

    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s' % ' '.join(arguments))
    process = subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        **kwargs)
    out, _ = process.communicate()
    return out, process.returncode

def run_command_background(arguments, log_file, **kwargs):
    
    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s' % ' '.join(arguments))
    process = subprocess.Popen(
        arguments,
        stdout = log_file,
        stderr = subprocess.STDOUT,
        **kwargs)
    return process

def command_check_call(arguments):

    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s', ' '.join(arguments))
    return subprocess.check_call(arguments)


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

def is_mounted(dev):
    """
    Check if the given device is mounted.
    """
    dev = os.path.realpath(dev)
    with file('/proc/mounts', 'rb') as proc_mounts:
        for line in proc_mounts:
            fields = line.split()
            if len(fields) < 3:
                continue
            mounts_dev = fields[0]
            path = fields[1]
            if mounts_dev.startswith('/') and os.path.exists(mounts_dev):
                mounts_dev = os.path.realpath(mounts_dev)
                if mounts_dev == dev:
                    return path
    return None

def is_held(dev):
    """
    Check if a device is held by another device (e.g., a dm-crypt mapping)
    """
    assert os.path.exists(dev)
    dev = os.path.realpath(dev)
    base = get_dev_name(dev)

    # full disk?
    directory = '/sys/block/{base}/holders'.format(base=base)
    if os.path.exists(directory):
        return os.listdir(directory)

    # partition?
    part = base
    while len(base):
        directory = '/sys/block/{base}/{part}/holders'.format(part=part, base=base)
        if os.path.exists(directory):
            return os.listdir(directory)
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

def list_partitions(basename):
    """
    Return a list of partitions on the given device name
    """
    partitions = []
    for name in os.listdir(os.path.join('/sys/block', basename)):
        if name.startswith(basename):
            partitions.append(name)
    return partitions

def is_partition(dev):
    """
    Check whether a given device path is a partition or a full disk.
    """
    dev = os.path.realpath(dev)
    if not stat.S_ISBLK(os.lstat(dev).st_mode):
        raise Error('not a block device', dev)

    name = get_dev_name(dev)
    if os.path.exists(os.path.join('/sys/block', name)):
        return False

    # make sure it is a partition of something else
    for basename in os.listdir('/sys/block'):
        if os.path.exists(os.path.join('/sys/block', basename, name)):
            return True

    raise Error('not a disk or partition', dev)

def device_not_in_use(dev, check_partitions = True):
    """
    Verify if a given device (path) is in use (e.g. mounted or
    in use by device-mapper).
    """
    assert os.path.exists(dev)

    if is_mounted(dev):
        return False
    holders = is_held(dev)

    if holders:
        return False

    if check_partitions and not is_partition(dev):
        basename = get_dev_name(os.path.realpath(dev))
        for partname in list_partitions(basename):
            partition = get_dev_path(partname)
            if is_mounted(partition):
                return False
            holders = is_held(partition)
            if holders:
                return False

    return True

# ---------- Own util stuff

def device_is_ssd(dev):
    """
    Check if given device is an ssd (device needs to be referenced as /dev/*)
    """
    dev = os.path.realpath(dev)
    if not stat.S_ISBLK(os.lstat(dev).st_mode):
        raise Error('not a block device', dev)

    dev_name = get_dev_name(dev)
    sys_path = '/sys/block/' + dev_name + '/queue/rotational'
    if not os.path.exists(sys_path):
       raise Error('Could not check if device "%s" is a ssd: path "%s" does not exist!', dev, sys_path)

    # Get info from sysfs
    rotational_raw = open(sys_path).read()

    if rotational_raw == '1':
        return False
    else:
        return True

def device_is_rotational(dev):
    return not device_is_ssd(dev)


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

def select_devices(args, remove_in_use = True):

    # List of selected devices referencing devices in /dev directly
    devices         = []

    # List of selected devices referencing devices by id (i.e. from /dev/disk/by-id)
    devices_by_id   = []

    LOG.info('Building device selection...')

    # Process selection by-id
    if args.disks_by_id:
        LOG.info('Adding device selection by id...')
        for device in os.listdir('/dev/disk/by-id'):

            # Ignore partitions
            if re.match(r'.*-part[0-9]+$', device):
                LOG.debug('Ignoring partition: %s', device)
                continue

            # Match devices
            if re.match(r'%s' % args.disks_by_id, device):
                LOG.info('Adding device to list: %s', device)
                devices.append(os.path.realpath('/dev/disk/by-id/' + device))

    # Process selection by-path
    if args.disks_by_path:
        LOG.info('Adding device selection by path...')
        for device in os.listdir('/dev/disk/by-path'):
            
            # Ignore partitions
            if re.match(r'.*-part[0-9]+$', device):
                LOG.debug('Ignoring partition: %s', device)
                continue

            # Match devices
            if re.match(r'%s' % args.disks_by_path, device):
                LOG.info('Adding device to list: %s', device)
                devices.append(os.path.realpath('/dev/disk/by-path/' + device))

    # Process selection by-partlabel
    if args.disks_by_partlabel:
        LOG.info('Adding device selection by partlabel...')
        for device in os.listdir('/dev/disk/by-partlabel'):
            
            # Match devices
            if re.match(r'%s' % args.disks_by_partlabel, device):
                LOG.info('Adding device to list: %s', device)
                devices.append(os.path.realpath('/dev/disk/by-partlabel/' + device))

    # Remove dupes
    devices = list(set(devices))

    # Translate to by-id devices for usage in commands

    # First build by-id -> /dev mapping from /dev/disk/by-id
    by_id_to_dev_map = {}
    for device in os.listdir('/dev/disk/by-id'):
            
        # Ignore partitions
        if re.match(r'.*-part[0-9]+$', device):
#            LOG.debug('Ignoring partition: %s', device)
            continue

        # Match devices
        if re.match(r'%s' % args.disks_by_id, device):
#            LOG.info('Adding device to list: %s', device)
            by_id_to_dev_map[device] = os.path.realpath('/dev/disk/by-id/' + device)
            LOG.debug("Mapping device '%s' to '%s'", device, by_id_to_dev_map[device])

    # Create inverse map
    pp = pprint.PrettyPrinter(indent=4)
    LOG.debug('by_id_to_dev_map: %s', pp.pprint(by_id_to_dev_map))

    dev_to_by_id_map = {v: k for k, v in by_id_to_dev_map.items()}
    LOG.debug('dev_to_by_id_map: %s', pp.pprint(dev_to_by_id_map))
 
    # Translate devices into by-id format (absolute path included)
    for device in devices:
        if remove_in_use:
            if device_not_in_use(device):
                devices_by_id.append('/dev/disk/by-id/' + dev_to_by_id_map[device])
            else:
                # Do not consider devices which are currently in use
                LOG.debug('Ignoring device "%s" as it is currently is use', dev_to_by_id_map[device])

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

    # Create a list of selected devices
    devices_by_id = select_devices(args)

    # Dump selected disks
    LOG.info('Selected disks:')
    for device in devices_by_id:
        LOG.info('- %s', device)

def main_badblocks(args):

    # Create log dir for badblocks
    mkdir(LOG_DIR + '/badblocks')

    # Create a list of devices to run badblocks on
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
        action  = 'store_true', default = None,
        help    = 'be more verbose', )

    parser.add_argument(
        '--logdir',
        metavar = 'PATH',
        default = '/tmp',
        help    = 'write log / output files to this dir (default /tmp)', )

    parser.add_argument(
        '--disks-by-id',
        metavar = 'DISKS',
        default = '',
        help    = 'Select disks by id via regexp against /dev/disk/by-id (default none)', )

    parser.add_argument(
        '--disks-by-path',
        metavar = 'DISKS',
        default = '',
        help    = 'Select disks by path via regexp against /dev/disk/by-path (default none)', )

    parser.add_argument(
        '--disks-by-partlabel',
        metavar = 'DISKS',
        default = '',
        help    = 'Select disks by partlabel via regexp against /dev/disk/by-partlabel (default none)', )

    parser.set_defaults(
        # we want to hold on to this, for later
        prog=parser.prog,
#        cluster='ceph', 
    )

    subparsers = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        help='sub-command help', )

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

    args = parser.parse_args()
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


