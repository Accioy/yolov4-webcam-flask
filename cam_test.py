import http.client
from urllib import parse


class MJPEGClient:
    """An iterable producing JPEG-encoded video frames from an MJPEG stream URL."""

    def __init__(self, url):
        self._url = parse.urlparse(url)

        h = http.client.HTTPConnection(self._url.netloc)
        h.request('GET', self._url.path)
        # h.endheaders()
        res = h.getresponse()

        if res.status == 200:
            self._fh = res
        else:
            raise RuntimeError()

    def __iter__(self):
        """Yields JPEG-encoded video frames."""

        while True:
            length = None
            while True:
                line = self._fh.readline()

                if line.startswith(b'Content-Length: '):
                    length = int(line.decode(encoding='utf-8').split(" ")[1])
                # Look for an empty line, signifying the end of the headers.
                if length is not None and line == b'\r\n':
                    break
            yield self._fh.read(length)
