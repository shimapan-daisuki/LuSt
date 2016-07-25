import sys
import os
import time


class ConsolePrinter(object):
    def __init__(self, _view, _fps_lock=1, _clear=False, _values_dictionary=None, _dest=sys.stdout.write, _mute=False,
                 _disabled=False, _flush=sys.stdout.flush, **kwargs):
        if _clear:
            self.clear = 'cls' if os.name == 'nt' else 'clear'
        else:
            self.clear = None
        if _fps_lock:
            self.delta = 1.0/float(_fps_lock)
        else:
            self.delta = None
        self.values_dictionary = dict()
        self.updates = dict()

        self.need_display = True
        self.last_display = time.time() - self.delta
        if kwargs:
            self.values_dictionary.update(**kwargs)
        if _values_dictionary:
            self.values_dictionary.update(_values_dictionary)
        self.dest = _dest
        self.flush = _flush
        self.view = _view
        self.mute = _mute
        self.disabled = _disabled

    def update(self, _updated_values=None, **kwargs):
        if not self.disabled:
            if _updated_values is None and not kwargs:
                return False
            if _updated_values:
                self.updates.update(_updated_values)
            if kwargs:
                self.updates.update(**kwargs)
            return True
        else:
            pass

    def display(self, _force=False, _updated_values=None, **kwargs):
        if not (self.mute or self.disabled):
            if self.update(_updated_values=_updated_values, **kwargs):
                self.values_dictionary.update(self.updates)
                self.need_display = True

            current_time = time.time()
            if _force or self.delta is None or (current_time - self.last_display >= self.delta and self.need_display):
                if self.clear:
                    os.system(self.clear)

                self.dest(self.view.format(**self.values_dictionary))
                if self.flush:
                    self.flush()

                self.updates.clear()
                self.need_display = False
                self.last_display = current_time
        else:
            pass

if __name__ == '__main__':
    pass

