class BaseConfig:
    """
    Shared Config between Components
    """

    pass


class ConfigLoaderMixin:
    """
    specify `config` in the class
    or overload the `get_config` method
    use `get_config`
    """

    config = None

    def get_config(self) -> BaseConfig:
        if self.config is None:
            raise Exception("Config not set")
        return self.config
