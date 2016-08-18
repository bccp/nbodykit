from .schema import ConstructorSchema 

import yaml
import traceback
from argparse import Namespace

class ConfigurationError(Exception): 
    """
    General exception for when parsing plugins from  fails
    """  
    pass
    
class EmptyConfigurationError(ConfigurationError):
    """
    Specific parsing error when the YAML loader does not
    find any valid keys
    """
    pass
    
class PluginParsingError(ConfigurationError):   
    """
    Specific parsing error when the plugin fails to load from 
    the configuration file
    """
    pass
    
def ReadConfigFile(stream, schema):
    """
    Read parameters from a file using YAML syntax
    
    The function uses the specified `schema` to:
    
        * check if parameter values are consistent with `choices`
        * infer the `type` of each parameter
        * check if any required parameters are missing
    
    Parameters
    ----------
    stream : open file object, str
        an open file object or the string returned by calling `read`
    schema : ConstructorSchema
        the schema which tells the parser which holds the relevant 
        information about the necessary parameters
    
    Returns
    -------
    ns : argparse.Namespace
        the namespace holding the loaded parameters that are valid
        according to the input schema
    unknown : argparse.Namespace
        a namespace holding any parsed parameters not present
        in the scema
    """
    from nbodykit.cosmology import Cosmology
    from nbodykit import plugin_manager, GlobalCosmology

    # make new namespaces to hold parsed attributes
    ns, unknown = Namespace(), Namespace()

    # read the yaml config file
    try:
        config = yaml.load(stream)
        
        # if the YAML loader returns a string or None
        # then it failed to find any valid keys
        if isinstance(config, (str, type(None))):
            raise EmptyConfigurationError("no valid keys found in input YAML file")
    
    except EmptyConfigurationError:
        raise # just re-raise this type of error
    except Exception as e:
        raise ConfigurationError("error parsing YAML file: %s" %str(e))
    
    # FIRST: tell plugin manager to load user plugins
    plugins = []
    if 'X' in config:
        plugins = config['X']
        if isinstance(plugins, str): 
            plugins = [plugins]
        plugin_manager.add_user_plugin(*plugins)
        config.pop('X')
    
    # SECOND: load any cosmology specified via 'cosmo' key
    cosmo = None
    if 'cosmo' in config:
       cosmo = Cosmology(**config.pop('cosmo'))
       GlobalCosmology.set(cosmo)
                
    # THIRD: iterate through schema, filling parameter namespace
    missing = []
    extra = config.copy()
    for name in schema:
        fill_namespace(ns, schema[name], extra, missing)
    
    # FOURTH: update the 'unknown' namespace with extra, ignored keys
    for k in extra:
        setattr(unknown, k, extra[k])
    
    # FIFTH: crash if we don't have all required parameters
    if len(missing):
        raise ValueError("missing required arguments: %s" %str(missing))
    
    return ns, unknown
 
def fill_namespace(ns, arg, config, missing):
    """
    Recursively fill the input namespace from a dictionary parsed
    from a YAML configuration file 
    
    Notes
    -----
    *   Fields that have sub-fields will be returned as sub-namespaces, such that
        the subfields can be accessed from the parent field with the same
        ``parent.subfield`` syntax
    *   Comparison of names between and configuration file and schema are 
        done in a case-insensitive manner
    *   Before adding to the namespace the values will be case according
        to the `cast` function specified via `arg`
    
    Parameters
    ----------
    ns : argparse.Namespace
        the namespace to fill with the loaded parameters 
    arg : Argument
        the schema Argument instance that we are trying to add to the 
        namespace; this holds the details about casting, sub-fields, etc
    config : dict
        a dictionary holding the parsed values from the YAML file
    missing : list
        a list to update with any arguments that are missing; 
        i.e., required by the schema but not present in `config`
    """
    # the name of the parameter (as taken from the schema)
    schema_name = arg.name.split('.')[-1]

    # no subfields
    if not len(arg.subfields):

        if config is not None:

            # the name of the parameter match in the configuration file
            # or None, if no match
            config_match = case_insensitive_name_match(schema_name, config)

            # parameter is present in the configuration dict
            if config_match is not None:
                value = config.pop(config_match)
                try:
                    setattr(ns, schema_name, ConstructorSchema.cast(arg, value))
                except Exception:
                    raise ConfigurationError("unable to cast '%s' value: %s" %(arg.name, traceback.format_exc()))
            else:
                # missing a required parameter
                if arg.required:
                  missing.append(arg.name)
    else:
        # recursively fill a sub namespace
        subns = Namespace()
        subconfig = config.pop(schema_name, None)

        for k in arg.subfields:
            fill_namespace(subns, arg[k], subconfig, missing)

        if len(vars(subns)):
            try:
                setattr(ns, schema_name, ConstructorSchema.cast(arg, subns))
            except Exception:
                raise ConfigurationError("unable to cast '%s' value: %s" %(arg.name, traceback.format_exc()))


def case_insensitive_name_match(schema_name, config):
    """
    Do case-insensitive name matching between the ConstructorSchema
    and the parsed configuration file
    
    Parameters
    ----------
    schema_name : str
        the name of the parameter, as given in the ConstructorSchema
    config : dict
        the parsed YAML dictionary from the configuration file
    
    Returns
    -------
    config_name : {str, None}
        return the key of `config` that matches `schema_name`; 
        otherwise, return `None`
    """
    # names from the parsed config file
    config_names = list(config.keys())
    lowered_config_names = [k.lower() for k in config_names]
    
    # lowered schema name
    lowered_schema_name = schema_name.lower()
    
    # return the name of the parameter in the configuration file
    if lowered_schema_name in lowered_config_names:
        index = lowered_config_names.index(lowered_schema_name)
        return config_names[index]
        
    return None