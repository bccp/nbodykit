class PluginMount(type):
    
    def __init__(cls, name, bases, attrs):

        # only executes when processing the mount point itself.
        if not hasattr(cls, 'plugins'):
            cls.plugins = []
        # called for each plugin, which already has 'plugins' list
        else:
            # track names of classes
            cls.plugins.append(cls)
            
            # try to call register class method
            if hasattr(cls, 'register'):
                cls.register()
            