def CheckXMLFile(f, header):
    """
    ATTRIBUTION: code provided by the OpenAI Gym package.

    Determines whether an XML file at given path is well-formed and contains specified header.

    :param f: string or os.path, absolute path to the xml file
    :param header: string, the header to be checked for
    :return: True if f points to a well-formed XML file with desired header, else False.
    """
    # Is f a file ?
    if not os.path.isfile(f):
        return False

    # Is f an XML file ?
    try:
        tree = et.parse(f)
    except et.ParseError:
        return False

    # Check the file header
    return tree.getroot().tag.upper() == header.upper()