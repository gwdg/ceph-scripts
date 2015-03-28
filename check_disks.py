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


# ---------- Own util stuff




processes = []

@atexit.register
def kill_subprocesses():
    for process in processes:
        process.kill()


DEVICES = '^(ata-HDS)|(ata-ST3000)'
DEVICES_PATH = '/dev/disk/by-id'

LOG_DIR = '/tmp'

THREADS = 30

BADBLOCKS_CALL = [ 'badblocks', '-b', '4096', '-v', '-w']

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

def main_badblocks(args):

    # Create log dir for badblocks
    mkdir(LOG_DIR + '/badblocks')

    pool = ThreadPool(THREADS)

    # Create a list of devices to run badblocks on
    devices = []
    for device in os.listdir(DEVICES_PATH):
        # Ignore partitions
        if re.match(r'.*-part[0-9]+$', device):
            LOG.debug('Ignoring partition: %s', device)
            continue
        # Match devices
        if re.match(r'%s' % DEVICES, device):
            LOG.info('Adding device to list: %s', device)
            devices.append(device)

    try:
        results = pool.map(check_disk, devices)

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
 
    parser.set_defaults(
        # we want to hold on to this, for later
        prog=parser.prog,
#        cluster='ceph',
        )

    subparsers = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        help='sub-command help',
        )

    # badblocks related arguments

    badblocks_parser = subparsers.add_parser('badblocks', help='Run badblocks for selection of disks')

    badblocks_parser.add_argument(
        '--cluster',
        metavar='NAME',
        help='cluster name to assign this disk to',)

    badblocks_parser.set_defaults(
        func=main_badblocks,
    )

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
        args.func(args)

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


