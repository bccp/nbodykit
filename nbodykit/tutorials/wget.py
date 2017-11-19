# Found from https://gist.github.com/remram44/6540454

from six.moves.html_parser import HTMLParser
from six.moves.urllib.error import HTTPError
from six.moves.urllib.request import urlopen
from six import string_types, PY2
import os
import re


# where the main nbodykit data examples live
data_url = "http://portal.nersc.gov/project/m779/nbodykit/example-data"

re_url = re.compile(r'^(([a-zA-Z_-]+)://([^/]+))(/.*)?$')

def resolve_link(link, url):
    m = re_url.match(link)
    if m is not None:
        if not m.group(4):
            # http://domain -> http://domain/
            return link + '/'
        else:
            return link
    elif link[0] == '/':
        # /some/path
        murl = re_url.match(url)
        return murl.group(1) + link
    else:
        # relative/path
        if url[-1] == '/':
            return url + link
        else:
            return url + '/' + link


class ListingParser(HTMLParser):
    """Parses an HTML file and build a list of links.

    Links are stored into the 'links' set. They are resolved into absolute
    links.
    """
    def __init__(self, url):
        HTMLParser.__init__(self)

        if url[-1] != '/':
            url += '/'
        self.__url = url
        self.links = set()

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for key, value in attrs:
                if key == 'href':
                    if not value:
                        continue
                    value = resolve_link(value, self.__url)
                    self.links.add(value)
                    break


def mirror(url, target=None):
    """
    Mirror a URL recursively to a local target.

    If ``target`` is not supplied, the last part of the url is used as
    the target.

    Parameters
    ----------
    url : str
        the URL to download
    target : str, optional
        the local file target to save the url to; if not provided, the
        last part of the url is used.
    """
    if target is None:
        target = os.path.normpath(url).split(os.path.sep)[-1]

    def mkdir():
        if not mkdir.done:
            try:
                os.mkdir(target)
            except OSError:
                pass
            mkdir.done = True
    mkdir.done = False

    # open the URL so we can parse it
    response = urlopen(url)

    # HTML file --> keep parsing
    info = response.info()
    content_type = info.type if PY2 else info.get_content_type()
    if content_type == 'text/html':
        contents = response.read().decode()

        parser = ListingParser(url)
        parser.feed(contents)
        for link in parser.links:
            link = resolve_link(link, url)
            if link[-1] == '/':
                link = link[:-1]
            if not link.startswith(url):
                continue
            name = link.rsplit('/', 1)[1]
            if '?' in name:
                continue
            mkdir()
            mirror(link, os.path.join(target, name))
        if not mkdir.done:
            # We didn't find anything to write inside this directory
            # Maybe it's a HTML file?
            if url[-1] != '/':
                end = target[-5:].lower()
                if not (end.endswith('.htm') or end.endswith('.html')):
                    target = target + '.html'
                with open(target, 'wb') as fp:
                    fp.write(contents.encode())
    # just download the file
    else:
        buffer_size = 4096*32
        with open(target, 'wb') as fp:
            chunk = response.read(buffer_size)
            while chunk:
                fp.write(chunk)
                chunk = response.read(buffer_size)

def available_examples():
    """
    Return a list of available example data files from the nbodykit
    data repository on NERSC.

    Returns
    -------
    examples : list
        list of the available file names for download
    """
    # read the contents of the main data URL
    response = urlopen(data_url)
    contents = response.read().decode()

    # parse the available files
    parser = ListingParser(data_url)
    parser.feed(contents)

    # get relative paths and remove bad links
    available = [os.path.relpath(link, data_url) for link in parser.links]
    available = [link for link in available if not any(link.startswith(bad) for bad in ['.', '?'])]
    return sorted(available)


def download_example_data(filenames, download_dirname=None):
    """
    Download a data file from the nbodykit repository of example data.

    For a list of valid file names, see :func:`available_examples`.

    Parameters
    ----------
    filenames : str, list of str
        the name(s) of the example file to download (relative to the path of the
        nbodykit repository); see :func:`available_examples` for the example
        file names
    download_dirname : str, optional
        a local directory to download the file to; if not specified, the
        file will be downloaded to the current working directory
    """
    if isinstance(filenames, string_types):
        filenames = [filenames]

    # make sure the download directory exists
    if download_dirname is not None:
        if not os.path.isdir(download_dirname):
            raise ValueError("specified download directory is not valid")

    # download all requested filenames
    for filename in filenames:

        # where we are saving locally
        if download_dirname is not None:
            target = os.path.join(download_dirname, filename)
        else:
            target = None

        # the full url to the data we want
        url = os.path.join(data_url, filename)

        # try to mirror locally
        try:
            mirror(url, target=target)
        except HTTPError as err:

            # if not found, print available file names, else just raise
            if err.code == 404:
                args = (filename, str(available_examples()))
                raise ValueError("no such example file '%s'\n\navailable examples are: %s" % args)
            else:
                raise
