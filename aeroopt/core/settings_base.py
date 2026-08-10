'''
Shared machinery for settings objects.

Every settings class can be built in two equivalent ways:

**From a JSON file** --- entries are matched by ``type`` (the class name) and
``name`` (the string passed to the constructor):

.. code-block:: python

    data_settings = SettingsData('wing_data', fname_settings='settings.json')

**From Python** --- either a mapping or keyword arguments, with no file
involved:

.. code-block:: python

    data_settings = SettingsData('wing_data', settings={'name_input': ['x1'], ...})

    data_settings = SettingsData.from_values(
        'wing_data',
        name_input=['x1'],
        input_low=[0.0],
        ...
    )

Both paths run the same conversion, defaulting and validation, so a study can
be defined inline for a quick experiment and exported to JSON later (see
:meth:`SettingsBase.to_dict` and :func:`save_settings`).

A subclass declares its fields once in ``_FIELDS``; it does not implement
loading.
'''

from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

#: JSON entry metadata; carried in ``settings`` but never applied as a field.
METADATA_KEYS = frozenset({'type', 'name'})

class _RequiredSentinel:
    '''
    Type of :data:`REQUIRED`; exists so the sentinel reprs readably in errors.
    '''
    def __repr__(self) -> str:
        return '<REQUIRED>'


#: Use as a field's default to mark the key as mandatory.
REQUIRED = _RequiredSentinel()

#: Deprecated private alias of :data:`REQUIRED`.
_REQUIRED = REQUIRED

#: Field specification: ``(attribute name, converter, default)``.
FieldSpec = Tuple[str, Callable[[Any], Any], Any]


class SettingsBase:
    '''
    Base class for settings objects.

    Parameters:
    -----------
    name: str
        Name of the settings entry. When loading from a file it selects the
        entry; when building from Python it is only a label.
    fname_settings: str
        Path to the settings file. Ignored when `settings` is given.
    settings: Mapping|None
        Field values to use instead of reading a file. Keys not declared in
        ``_FIELDS`` are ignored unless ``_ALLOW_EXTRA_KEYS`` is set.

    Attributes:
    -----------
    name: str
        Name of the settings entry.
    settings: Dict[str, Any]
        The raw mapping this object was built from.
    '''

    #: Field specifications, see :data:`FieldSpec`. Overridden by subclasses.
    _FIELDS: Tuple[FieldSpec, ...] = ()

    #: When True, keys that are not in ``_FIELDS`` are set verbatim as
    #: attributes. Used by :class:`SettingsOptimization` for forward
    #: compatibility with user-defined keys.
    _ALLOW_EXTRA_KEYS: bool = False

    #: Print a line when an entry is read from a file.
    _VERBOSE_FILE_READ: bool = True

    def __init__(self, name: str,
                 fname_settings: str = 'settings.json',
                 *,
                 settings: Mapping[str, Any] | None = None):

        self.name = name
        self.settings: Dict[str, Any] = {}

        self._apply_defaults()

        if settings is None:
            entry = self._find_settings_entry(fname_settings)
        else:
            entry = dict(settings)

        self._apply_entry(entry, origin=fname_settings if settings is None else '<python>')

    #* Construction

    @classmethod
    def from_values(cls, name: str, *args: Any, **fields: Any):
        '''
        Build settings from keyword arguments, without a settings file.

        Positional arguments after `name` are forwarded to the constructor, so
        classes that need extra objects still work::

            SettingsData.from_values('d', name_input=['x1'], input_low=[0.0], ...)
            SettingsProblem.from_values('p', data_settings, output_type=[-1])

        Parameters:
        -----------
        name: str
            Name of the settings entry.
        \\*args
            Extra positional constructor arguments, if the subclass takes any.
        \\*\\*fields
            Field values, matching the keys a JSON entry would hold.

        Returns:
        --------
        settings: SettingsBase
            The constructed settings object.
        '''
        return cls(name, *args, settings=fields)

    #* Field handling

    @classmethod
    def _entry_type(cls) -> str:
        '''
        The ``type`` value identifying this class's entries in a JSON file.
        '''
        return cls.__name__

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        '''
        Names of the declared fields, in declaration order.
        '''
        return tuple(attribute for attribute, _convert, _default in cls._FIELDS)

    def _apply_defaults(self) -> None:
        '''
        Set every optional field to its default, so an attribute always exists.
        '''
        for attribute, _convert, default in self._FIELDS:
            if default is _REQUIRED:
                continue
            # Copy containers: a shared default would be mutated by one
            # instance and observed by every other.
            if isinstance(default, (list, dict, set)):
                default = copy.copy(default)
            setattr(self, attribute, default)

    def _apply_entry(self, entry: Mapping[str, Any], origin: str) -> None:
        '''
        Convert and assign the fields of one settings entry.
        '''
        self.settings = dict(entry)

        declared = set(self.field_names())

        for attribute, convert, default in self._FIELDS:
            if attribute in entry:
                setattr(self, attribute, convert(entry[attribute]))
            elif default is _REQUIRED:
                raise ValueError(
                    f'{self._entry_type()} {self.name}: required key '
                    f'"{attribute}" is missing in {origin}.')

        if self._ALLOW_EXTRA_KEYS:
            for key, value in entry.items():
                if key in METADATA_KEYS or key in declared:
                    continue
                setattr(self, key, value)

    def _find_settings_entry(self, fname_settings: str) -> Mapping[str, Any]:
        '''
        Locate the JSON entry whose ``type`` matches this class and whose
        ``name`` matches ``self.name``.

        Raises
        ------
        FileNotFoundError
            When the settings file does not exist.
        ValueError
            When no entry matches.
        '''
        if not os.path.exists(fname_settings):
            raise FileNotFoundError(f'Settings file {fname_settings} not found.')

        with open(fname_settings, encoding='utf-8') as f:
            settings = json.load(f)

        entry_type = self._entry_type()

        matched = None
        for entry_name, entry_data in settings.items():
            if entry_data.get('type') == entry_type and entry_data.get('name') == self.name:
                if self._VERBOSE_FILE_READ:
                    print(f'>>> {entry_type} {self.name} ({entry_name}) read successfully.')
                matched = entry_data

        if matched is None:
            raise ValueError(f'{entry_type} {self.name} not found in {fname_settings}.')

        return matched

    def read_settings(self, fname_settings: str) -> None:
        '''
        Reload the fields of this object from a settings file.
        '''
        self._apply_entry(self._find_settings_entry(fname_settings), fname_settings)

    #* Export

    def to_dict(self) -> Dict[str, Any]:
        '''
        Serialize this object as a JSON-ready settings entry.

        The result includes the ``type`` and ``name`` metadata, so it can be
        written straight into a settings file and read back by the constructor.
        NumPy arrays become lists.

        Returns:
        --------
        entry: Dict[str, Any]
            The settings entry.
        '''
        entry: Dict[str, Any] = {'type': self._entry_type(), 'name': self.name}

        for attribute in self.field_names():
            value = getattr(self, attribute, None)
            entry[attribute] = _to_jsonable(value)

        return entry


def _to_jsonable(value: Any) -> Any:
    '''
    Convert numpy scalars and arrays to plain Python for JSON serialization.
    '''
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_settings(settings_objects: Iterable[SettingsBase], fname_settings: str,
                  entry_names: Iterable[str] | None = None) -> None:
    '''
    Write settings objects to a JSON file that the constructors can read back.

    This is the bridge from a Python-defined study to a shareable,
    version-controllable settings file.

    Parameters:
    -----------
    settings_objects: Iterable[SettingsBase]
        The objects to serialize.
    fname_settings: str
        Path of the JSON file to write.
    entry_names: Iterable[str]|None
        Top-level keys for each entry. Defaults to ``<type>_<name>``.

    Example:
    ---------
    >>> save_settings([data_settings, problem_settings, opt_settings],
    ...               'settings.json')
    '''
    objects = list(settings_objects)

    if entry_names is None:
        keys = [f'{obj._entry_type()}_{obj.name}' for obj in objects]
    else:
        keys = list(entry_names)
        if len(keys) != len(objects):
            raise ValueError(
                f'entry_names has {len(keys)} names for {len(objects)} objects.')

    directory = os.path.dirname(os.path.abspath(fname_settings))
    os.makedirs(directory, exist_ok=True)

    document = {key: obj.to_dict() for key, obj in zip(keys, objects)}

    with open(fname_settings, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=4, ensure_ascii=False)
        f.write('\n')
