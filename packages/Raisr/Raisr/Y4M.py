# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from collections import namedtuple


class Frame(namedtuple('Frame', ['y', 'uv', 'headers','binary_headers', 'count'])):
    def __repr__(self):
        return '<frame %d: %dx%d>' % (self.count, self.headers['H'], self.headers['W'])

class Reader(object):
    def __init__(self, callback, verbose=False, preprocessed=False):
        self._callback = callback
        self._stream_headers = None
        self._data = bytes()
        self._count = 0
        self._verbose = verbose
        self._preprocessed = preprocessed

    def _print(self, *args):
        if self._verbose:
            print('Y4M Reader:', ' '.join([str(e) for e in args]))

    def decode(self, data):
        assert isinstance(data, bytes)
        self._data += data
        if self._stream_headers is None:
            self._decode_stream_headers()
            if self._stream_headers is not None:
                self._print('detected stream with headers:', self._stream_headers)
        if self._stream_headers is not None:
            frame = self._decode_frame()
            while frame is not None:
                self._print(frame, 'decoded')
                #self._callback(frame)
                #frame = self._decode_frame()
                return frame

    def _frame_size(self):
        assert self._stream_headers['C'].startswith('420'), 'only support I420 fourcc'
        if self._stream_headers['C'] == '420p10':
            return self._stream_headers['W'] * self._stream_headers['H'] * 2 * 3 // 2
        return self._stream_headers['W'] * self._stream_headers['H'] * 3 // 2

    def _decode_frame(self):
        # Spend most of your time in this method. Use the debugger and check the values in toks
        if len(self._data) < self._frame_size():  # no point trying to parse
            return None
        if self._preprocessed:
            toks = [0] * 2
            toks[0] = b'FRAME'
            if self._data.startswith(b'FRAME'):
                toks[1] = self._data[len(toks[0]) + 1:len(toks[0]) + 1 + self._frame_size()]
            else:
                toks[1] = self._data[:self._frame_size()]
        else:
            toks = self._data.split(b'\n',1)

        if len(toks) == 1:  # need more data
            self._print('weird: got plenty of data but no frame header found')
            return None
        headers = toks[0].split(b' ')
        #assert headers[0] == b'FRAME', 'expected FRAME (got %r)' % headers[0]
        frame_headers = self._stream_headers.copy()
        for header in headers[1:]:
            header = header.decode('ascii')
            frame_headers[header[0]] = header[1:]
        if len(toks[1]) < self._frame_size():  # need more data
            return None
        ysize = self._frame_size() * 2 // 3
        y = toks[1][0:ysize]
        uv = toks[1][ysize:self._frame_size()]
        # padding = 6 if not self._preprocessed else 0
        # self._data = self._data[self._frame_size() + padding]
        self._data = toks[1][self._frame_size():]
        self._count += 1
        return Frame(y, uv, frame_headers, self._binary_headers, self._count - 1)

    def _decode_stream_headers(self):
        toks = self._data.split(b'\n', 1)
        if len(toks) == 1:  # buffer all header data until eof
            return
        self._stream_headers = {}
        self._binary_headers = toks[0]
        self._data = toks[1]  # save the beginning of the stream for later
        headers = toks[0].split(b' ')
        if headers[0] != b'YUV4MPEG2':
            self._convert2y4m(toks[0])
            self._decode_stream_headers()
            return
        assert headers[0] == b'YUV4MPEG2', 'unknown type %s' % headers[0]
        for header in headers[1:]:
            header = header.decode('ascii')
            self._stream_headers[header[0]] = header[1:]
        assert 'W' in self._stream_headers, 'No width header'
        assert 'H' in self._stream_headers, 'No height header'
        assert 'F' in self._stream_headers, 'No frame-rate header'
        self._stream_headers['W'] = int(self._stream_headers['W'])
        self._stream_headers['H'] = int(self._stream_headers['H'])
        self._stream_headers['F'] = [int(n) for n in self._stream_headers['F'].split(':')]
        if 'A' in self._stream_headers:
            self._stream_headers['A'] = [int(n) for n in self._stream_headers['A'].split(':')]
        if 'C' not in self._stream_headers:
            self._stream_headers['C'] = '420jpeg'  # man yuv4mpeg

    def set_headers(self, headers):
        self._stream_headers = {}
        headers = headers.split(b' ')
        assert headers[0] == b'YUV4MPEG2', 'unknown type %s' % headers[0]
        for header in headers[1:]:
            header = header.decode('ascii')
            self._stream_headers[header[0]] = header[1:]
        assert 'W' in self._stream_headers, 'No width header'
        assert 'H' in self._stream_headers, 'No height header'
        assert 'F' in self._stream_headers, 'No frame-rate header'
        self._stream_headers['W'] = int(self._stream_headers['W'])
        self._stream_headers['H'] = int(self._stream_headers['H'])
        self._stream_headers['F'] = [int(n) for n in self._stream_headers['F'].split(':')]
        if 'A' in self._stream_headers:
            self._stream_headers['A'] = [int(n) for n in self._stream_headers['A'].split(':')]
        if 'C' not in self._stream_headers:
            self._stream_headers['C'] = '420jpeg'  # man yuv4mpeg

    def _convert2y4m(self, headers):
        import ffmpeg
        try:
            out, err = (
                ffmpeg
                .input('pipe:')
                .output('pipe:', **{'f':'yuv4mpegpipe','pix_fmt':'yuv420p10le','strict':-1})
                .run(quiet=True,input=b'\n'.join([headers,self._data]))
            )
            self._data = out
        except ffmpeg.Error as e:
            print(e.stderr)


class Writer(object):
    def __init__(self, fd, verbose=False):
        self._fd = fd
        self._stream_headers = None
        self._count = 0
        self._verbose = verbose

    def _print(self, *args):
        if self._verbose:
            print('Y4M Writer:', ' '.join([str(e) for e in args]))

    def encode(self, frame):
        assert isinstance(frame, Frame), 'only Frame object are supported'
        if self._stream_headers is None:
            assert 'W' in frame.headers, 'No width header'
            assert 'H' in frame.headers, 'No height header'
            assert 'F' in frame.headers, 'No frame-rate header'
            self._stream_headers = frame.headers.copy()
            if 'C' not in self._stream_headers:
                self._stream_headers['C'] = '420jpeg'  # man yuv4mpeg
            data = self._encode_headers(self._stream_headers.copy())
            self._fd.write(b'YUV4MPEG2 ' + data + b'\n')
            self._print('generating stream with headers:', self._stream_headers)
        self._encode_frame(frame)

    def _frame_size(self):
        assert self._stream_headers['C'].startswith('420'), 'only support I420 fourcc'
        if self._stream_headers['C'] == '420p10':
            return self._stream_headers['W'] * self._stream_headers['H'] * 2 * 3 // 2
        return self._stream_headers['W'] * self._stream_headers['H'] * 3 // 2

    def _encode_headers(self, headers):
        for k in headers.keys():
            if isinstance(headers[k], int):
                headers[k] = str(headers[k])
            elif isinstance(headers[k], list):
                headers[k] = ':'.join([str(i) for i in headers[k]])
        data = b' '.join([k.encode('ascii') + v.encode('ascii') for k, v in headers.items()])
        return data

    def _encode_frame(self, frame):
        assert len(frame.y) + len(frame.uv) == self._frame_size()
        #data = self._encode_headers(frame.headers)
        self._fd.write(b'FRAME'b'\n')
        self._fd.write(frame.y)
        self._fd.write(frame.uv)