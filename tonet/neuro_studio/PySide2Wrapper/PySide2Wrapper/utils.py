import json
import os


class StateSaver:
    """
    StateSaver is a ui state manager. This works with Window and store it's widget states in file.
    That may used for restore ui state between process starts.
    """
    def __init__(self, store_path: str):
        self.__path = store_path
        self.__is_loaded = False
        self.__widgets = []

    def add_widget(self, widget):
        """
        Add widget to StateSaver
        :param widget: widget object
        """
        self.__widgets.append(widget)

    def write(self):
        """
        Write all states of all widgets to file
        """
        data = {i: w for i, w in enumerate(self.__widgets) if not (type(w.get_value()) is str and w.get_value() == "") and w.get_value() is not None}
        with open(self.__path, 'w') as outfile:
            json.dump(data, outfile, default=lambda w: w.get_value())

    def load(self):
        """
        Load all states of all widgets from file
        """
        if self.__is_loaded or (not (os.path.exists(self.__path) and os.path.isfile(self.__path))):
            return

        try:
            with open(self.__path, 'r') as infile:
                states = json.load(infile)

            for k, v in states.items():
                self.__widgets[int(k)].set_value(v)
        except:
            os.remove(self.__path)

        self.__is_loaded = True
