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
from multiprocessing.dummy import Pool as ThreadPool
from Queue import PriorityQueue

# Initialize logging

# Create logger
LOG = logging.getLogger('check_disks')
LOG.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
LOG.addHandler(ch)

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

def run_command_background(arguments, **kwargs):
    
    arguments = _get_command_executable(arguments)
    LOG.info('Running command: %s' % ' '.join(arguments))
    process = subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
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

DEVICES = '^(ata-HDS)|(ata-ST3000)'
DEVICES_PATH = '/dev/disk/by-id'

LOG_DIR = '/tmp/badblocks'

THREADS = 30

BADBLOCKS_CALL = [ 'badblocks', '-b', '4096', '-v', '-w']

def check_disk(device):

    device_path = DEVICES_PATH + '/' + device
    LOG.info('Running badblocks on device: %s', device_path)

#    if not stat.S_ISBLK(os.lstat(device).st_mode):
#        raise Error('not a block device', device)
    
    log_file = LOG_DIR + '/' + device + '.txt'
    LOG.debug('Using log file for badblocks data: %s', log_file)

    badblocks_call = BADBLOCKS_CALL[:]
    badblocks_call.extend(['-o', log_file, device_path])

    process = run_command_background(badblocks_call)
    processes.append(process)

    out, _ = process.communicate()

    processes.remove(process)

# Create log dir for badblocks
mkdir(LOG_DIR)

pool = ThreadPool(THREADS)

# Create a list of devices to run badblocks on
devices = []
for device in os.listdir(DEVICES_PATH):
    # Ignore partitions
    if re.match(r'.*-part[0-9]+$', device):
        LOG.debug('Ignoring partition: %s', device)
        continue
    if re.match(r'%s' % DEVICES, device):
        LOG.info('Adding device to list: %s', device)
        devices.append(device)

processes = []

@atexit.register
def kill_subprocesses():
    for process in processes:
        process.kill()

try:
    results = pool.map(check_disk, devices)
except KeyboardInterrupt:
    kill_subprocesses()
    sys.exit()

pool.close() 
pool.join() 

