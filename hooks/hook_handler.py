class HookHandler:
    def __init__(self) -> None:
        self.hooks = []

    def register(self, new_hook):
        self.hooks.append(new_hook)

    def call(self, method):
        for hook in self.hooks:
            if hasattr(hook, method) and callable(fun := getattr(hook, method)):
                fun()

    