from fastcore.utils import is_listy


class Hook:
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m, hook_func, is_forward=True, detach=True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (
                (o.detach() for o in output) if is_listy(output) else output.detach()
            )
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


# Cell
class Hooks:
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward=True, detach=True):
        for k, m in enumerate(ms):
            setattr(self, f"hook_{k}", Hook(m, hook_func, is_forward, detach))
        # self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]
        self.n = len(ms)

    def __getitem__(self, i):
        return getattr(self, f"hook_{i}")

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter([self[k] for k in range(len(self))])

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()
