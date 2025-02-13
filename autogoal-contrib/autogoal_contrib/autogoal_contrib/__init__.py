from typing import List, Tuple
from autogoal.kb import AlgorithmBase
from autogoal.utils import find_packages
from os.path import abspath, dirname, join
from inspect import getsourcefile
from autogoal_contrib.wrappers import *
import enum
import yaml

config_dir = dirname(abspath(getsourcefile(lambda: 0)))


def find_classes(include=None, exclude=None, modules=None, input=None, output=None, include_classes=None, exclude_classes=None):
    """
    This function finds and returns algorithm classes based on the provided criteria.

    Parameters:
    - include (str, optional): A regular expression pattern. Only classes whose string representation matches this pattern will be included.
    - exclude (str, optional): A regular expression pattern. Classes whose string representation matches this pattern will be excluded.
    - modules (list, optional): A list of modules to search for classes. If not provided, the function will search in all installed contribs excluding "remote".
    - input (str, optional): A regular expression pattern. Only classes whose 'run' method's 'input' parameter annotation matches this pattern will be included.
    - output (str, optional): A regular expression pattern. Only classes whose 'run' method's return annotation matches this pattern will be included.
    - include_classes (list, optional): A list of specific classes to include.
    - exclude_classes (list, optional): A list of specific classes to exclude.

    Returns:
    - list: A list of classes that match the provided criteria.

    The function searches for classes in the specified modules. It includes a class if it has a 'run' method, its name does not start with '_', '_builder' is not in its module name, and it is a subclass of AlgorithmBase but not AlgorithmBase itself. The function also checks the 'include', 'exclude', 'input', and 'output' criteria if they are provided. Finally, it adds the classes in 'include_classes' and removes the classes in 'exclude_classes'.
    """
    
    import inspect
    import re

    result = []
    
    include_classes = set() if include_classes == None else set(include_classes)      
    exclude_classes = set() if exclude_classes == None else set(exclude_classes)      

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if input:
        input = f".*({input}).*"

    if output:
        output = f".*({output}).*"

    if modules is None:
        modules = []

        for module in get_installed_contribs(exclude=["remote"]):
            modules.append(module)

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not hasattr(cls, "run"):
                continue

            if cls.__name__.startswith("_"):
                continue

            if "_builder" in cls.__module__:
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue
            
            if cls in exclude_classes:
                continue

            if not issubclass(cls, AlgorithmBase) or cls is AlgorithmBase:
                continue

            sig = inspect.signature(cls.run)

            if input and not re.match(input, str(sig.parameters["input"].annotation)):
                continue

            if output and not re.match(output, str(sig.return_annotation)):
                continue

            result.append(cls)

    # add Included classes
    for cls in include_classes:
        if not cls in result:
            result.append(cls)

    return result


def find_remote_classes(
    sources: List[Tuple[str, int] or Tuple[str, int, str] or str] = None,
    include=None,
    exclude=None,
    input=None,
    output=None,
    ignore_cache=False,
):
    """
    This function finds and returns remote classes based on the provided criteria.

    Parameters:
    - sources (list, optional): A list of sources to search for classes. Each source can be a string (alias), a tuple of two elements (IP and port), or a tuple of three elements (IP, port, and alias). If not provided, the function will search in all stored aliases.
    - include (str, optional): A regular expression pattern. Only classes whose string representation matches this pattern will be included.
    - exclude (str, optional): A regular expression pattern. Classes whose string representation matches this pattern will be excluded.
    - input (str, optional): A regular expression pattern. Only classes whose 'input_types' method's return value matches this pattern will be included.
    - output (str, optional): A regular expression pattern. Only classes whose 'output_type' method's return value matches this pattern will be included.
    - ignore_cache (bool, optional): If True, the function will ignore the cache and fetch the algorithms from the remote server.

    Returns:
    - list: A list of remote classes that match the provided criteria.

    The function searches for classes in the specified sources. It includes a class if its string representation matches the 'include' pattern, its 'input_types' method's return value matches the 'input' pattern, and its 'output_type' method's return value matches the 'output' pattern. The function excludes a class if its string representation matches the 'exclude' pattern.
    """
    from autogoal_remote import get_algorithms, store_connection, get_stored_aliases
    import itertools
    import re

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if input:
        input = f".*({input}).*"

    if output:
        output = f".*({output}).*"

    if len(sources) == 0:
        sources = [(alias.ip, alias.port) for alias in get_stored_aliases()]

    classes_by_contrib = {}
    for source in sources:
        s_type = type(source)
        ip = None
        port = None
        alias = source if s_type == str else None

        if s_type == tuple:
            if len(source) == 2:
                ip, port = source
            elif len(source) == 3:
                ip, port, alias = source
                store_connection(ip, port, alias)

        for contrib in itertools.groupby(
            get_algorithms(ip, port, alias), lambda x: x.contrib
        ):
            key, classes = contrib
            contrib_results = []

            for cls in classes:
                try:
                    if not re.match(include, repr(cls)):
                        continue

                    if exclude is not None and re.match(exclude, repr(cls)):
                        continue

                    inp = cls.input_types()
                    outp = cls.output_type()

                    if input and not re.match(input, str(inp)):
                        continue

                    if output and not re.match(output, str(outp)):
                        continue

                    contrib_results.append(cls)
                except:
                    pass

            classes_by_contrib[key] = contrib_results

    result = [cls for _, classes in classes_by_contrib.items() for cls in classes]
    return result


def resolve_algorithm(cls_name: str):
    """
    Returns the first Algorithm definition from a local or remote contrib source that matches the `cls_name`.
    Returns `None` if no matching Algorithm was found.
    """
    classes = []
    classes += find_classes()
    classes += find_remote_classes()

    for cls in classes:
        if cls.__name__ == cls_name:
            return cls


def get_registered_contribs():
    path = join(config_dir, "registered-contribs.yml")
    try:
        with open(path, "r") as fd:
            result = yaml.safe_load(fd)
    except IOError as e:
        result = []
    return result


def get_installed_contribs(exclude: List[str] = None):
    """
    find all installed contrib modules.
    """

    exclude_pattern = "" if exclude is None else rf"(?!{'|'.join(exclude)})"
    packages_identifier = rf"autogoal-{exclude_pattern}.*"
    modules = []
    for pkg in find_packages(packages_identifier):
        try:
            key = pkg.key.replace("-", "_")
            module = __import__(key)
            modules.append(module)
        except ImportError as e:
            print(
                f"Error importing {pkg.project_name} {pkg.version}. Use pip install {pkg.project_name} to ensure all dependencies are installed correctly."
            )
    return modules


class ContribStatus(enum.Enum):
    RequiresDependency = enum.auto()
    RequiresDownload = enum.auto()
    Ready = enum.auto()


def status():
    status = {}
    modules = []

    for pkg_name in get_registered_contribs():
        status[pkg_name] = ContribStatus.RequiresDependency

    for module in get_installed_contribs():
        status[module.project_name] = ContribStatus.Ready
        modules.append(module)

    modules.sort(key=lambda m: m.project_name)
    for module in modules:
        if hasattr(module, "status"):
            status[module.project_name] = module.status()

    return status


def download(contrib: str):
    modules = {}

    for package in get_installed_contribs():
        modules[package.key] = __import__(package.key)

    if contrib not in modules:
        raise ValueError(f"Contrib `{contrib}` cannot be imported.")

    contrib = modules[contrib]

    if not hasattr(contrib, "download"):
        return False

    return contrib.download()


__all__ = [
    "find_classes",
    "status",
    "download",
    "get_installed_contribs",
    "get_registered_contribs",
]

if __name__ == "__main__":
    # print(find_classes())
    # print(get_installed_contribs(exclude = "remote"))
    print(find_remote_classes(["remote-sklearn"]))
