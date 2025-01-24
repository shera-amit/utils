import numpy as np
import pandas as pd
from collections import defaultdict
import re
import os
import tempfile


class LogBase:
    def __init__(self):
        self.localFiles = []
        self.remoteFiles = []

    def isRemoteFile(self, fname):
        return fname.lower().startswith('ssh:') or fname.lower().startswith('sftp:')

    def dowloadRemote(self, fname):
        import subprocess as sub
        # remote path defined as ssh:server:path or sftp:server:path
        _, server, remoteFile = fname.split(':')
        localFile = os.path.join(
            tempfile.gettempdir(), str(abs(hash(remoteFile))))
        _ = sub.check_output(['scp', f'{server}:{remoteFile}', localFile])
        return localFile

    def checkFileName(self, fname):
        if fname in self.fileNames():
            raise ValueError('{} has already been parsed'.format(fname))

    def getFileName(self, fname):
        if self.isRemoteFile(fname):
            fname = self.dowloadRemote(fname)
            self.checkFileName(fname)
            self.remoteFiles.append(fname)
        else:
            self.checkFileName(fname)
            self.localFiles.append(fname)
        return fname

    def cleanupTempFile(self, fname):
        if (os.path.split(fname)[0] == tempfile.gettempdir()) and os.path.isfile(fname):
            os.remove(fname)

    def __del__(self):
        [self.cleanupTempFile(f) for f in self.remoteFiles]

class Log(LogBase):
    def __init__(self, fname=None):
        super().__init__()
        self.data = pd.DataFrame()
        self.runInfo = defaultdict(dict)
        self.run = 0
        if fname:
            self.parseLog(fname)

    def __del__(self):
        super().__del__()

    def __getitem__(self, arg):
        if isinstance(arg, str):
            if arg.lower() == 'index':
                return self.data.index
            elif arg.lower() == 'n':
                return self.getN()
            else:
                try:
                    return self.data[arg]
                except KeyError:
                    err = 'Key {} not found! Valid keys are: {}'.format(
                        arg, ' '.join(self.keys()))
                    raise KeyError(err)
        elif isinstance(arg, int):
            if arg in self.runInfo:
                return self.runInfo[arg]
            elif arg == -1:
                return self.runInfo[self.run-1]
            else:
                raise KeyError(
                    'Key {} not found! Valid numeric keys are:'.format(arg),
                    ' '.join([str(k) for k in self.runInfo.keys()]))
        else:
            raise KeyError('Key {} not found!'.format(arg))

    def getN(self):
        N = np.array([self.runInfo[i]['N'] for i in range(self.run)])
        if np.all(N == N[0]):
            return N[0]
        else:
            return N
    def N(self):
        return self.getN()

    def converged(self, arg):
        try:
            arg = int(arg)
        except ValueError:
            raise KeyError('Only numeric indicies are allowed for runs!')
        converged = ['force tolerance',
                     'energy tolerance',
                     'linesearch alpha is zero']
        try:
            return self[arg]['StoppingCriterion'] in converged
        except KeyError:
            return False

    def keys(self):
        return self.data.keys()

    def fileNames(self):
        return self.localFiles + self.remoteFiles

    def parseLog(self, fname):
        fname = self.getFileName(fname)
        newData = []
        headerline = False
        dataline = False
        tailline = False
        with open(fname, 'r') as f:
            for line in f:
                if (line[:12] == 'Memory usage') or \
                        (line[:12] == 'Per MPI rank'):
                    headerline = True
                    tailline = False
                    continue
                if line[:4] == 'Loop':
                    headerline = False
                    dataline = False
                    tailline = True
                    self.runInfo[self.run]['N'] = int(line.split()[-2])
                    self.runInfo[self.run]['LoopTime'] = float(line.split()[3])
                    self.run += 1
                if headerline:
                    headerline = False
                    dataline = True
                    categories = line.strip().split()
                    continue
                if (dataline) and not (headerline) and not (tailline):
                    try:
                        lines = [float(v) for v in line.strip().split()]
                    except ValueError:
                        continue
                    newData.append({c: v for c, v in zip(categories, lines)})
                    newData[-1]['Run'] = self.run
                if 'Stopping criterion' in line:
                    sc = line.split('=')[-1].strip()
                    self.runInfo[self.run-1]['StoppingCriterion'] = sc
                    self.runInfo[self.run-1]['Converged'] = self.converged(-1)
        self.data = self.data.append(newData, ignore_index=True)


class elasticConstantsLog:
    def __init__(self, fname=None):
        self.data = {}
        self.Cmat = np.zeros((6, 6))
        self.Smat = np.zeros((6, 6))
        if fname:
            self.parseLog(fname)

    def __getitem__(self, arg):
        try:
            if isinstance(arg, str):
                return self.data[arg.lower()]
            elif isinstance(arg, int):
                return self.data['c{}'.format(arg)]
            else:
                raise KeyError()
        except KeyError:
            err = 'Key {} not found! Valid keys are: {}'.format(
                arg, ' '.join(self.data.keys()))
            raise KeyError(err)

    def parseLog(self, fname):
        pattern = re.compile(
            '.* (C\d\d)all = (-?\d+\.?\d*e?-?\+?\d?\d?\d?) (\w+)')
        with open(fname, 'r') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    if 'unit' in self.data:
                        assert self.data['unit'] == match.groups()[2]
                    else:
                        self.data['unit'] = match.groups()[2]
                    self.data[match.groups()[0].lower()] = float(
                        match.groups()[1])
        if len(self.data) == 0:
            raise ValueError('{} is not a valid log file!'.format(fname))
        self._getVRH()

    def _createCmat(self):
        if len(self.data) == 0:
            raise ValueError('No log file loaded!')
        for i in range(0, self.Cmat.shape[0]):
            for j in range(i, self.Cmat.shape[1]):
                self.Cmat[i, j] = self['c{}{}'.format(i+1, j+1)]
                self.Cmat[j, i] = self.Cmat[i, j]
        try:
            self.Smat = np.linalg.inv(self.Cmat)
        except np.linalg.LinAlgError:
            self.Smat[:, :] = np.nan

    def _getVRH(self):
        if np.isclose(np.sum(self.Smat), 0) or np.isclose(np.sum(self.Cmat), 0):
            self._createCmat()
        self.data['kv'] = ((self.Cmat[0, 0]+self.Cmat[1, 1]+self.Cmat[2, 2]) +
                           2*(self.Cmat[0, 1]+self.Cmat[1, 2]+self.Cmat[2, 0]))/9
        self.data['gv'] = ((self.Cmat[0, 0]+self.Cmat[1, 1]+self.Cmat[2, 2]) -
                           (self.Cmat[0, 1]+self.Cmat[1, 2]+self.Cmat[2, 0]) +
                           3*(self.Cmat[3, 3]+self.Cmat[4, 4]+self.Cmat[5, 5]))/15
        self.data['kr'] = 1/((self.Smat[0, 0]+self.Smat[1, 1]+self.Smat[2, 2]) +
                             2*(self.Smat[0, 1]+self.Smat[1, 2]+self.Smat[2, 0]))
        self.data['gr'] = 15/(4*(self.Smat[0, 0]+self.Smat[1, 1]+self.Smat[2, 2]) -
                              4*(self.Smat[0, 1]+self.Smat[1, 2]+self.Smat[2, 0]) +
                              3*(self.Smat[3, 3]+self.Smat[4, 4]+self.Smat[5, 5]))
        self.data['g'] = (self.data['gr']+self.data['gv'])/2
        self.data['k'] = (self.data['kr']+self.data['kv'])/2
        self.data['nu'] = 0.5*(1-(3*self.data['g']) /
                               (3*self.data['k']+self.data['g']))
        self.data['e'] = 1/((1/3/self.data['g']) + (1/9/self.data['k']))
