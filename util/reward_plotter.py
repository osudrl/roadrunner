from multiprocessing import Process, Queue
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Plotter:
    """Plots env info metrics in real time. Uses multiprocessing to avoid blocking the main process."""
    def __init__(self):
        matplotlib.use('TkAgg')
        self.data_queue = Queue()
        self.plotting_process = Process(target=self._plot_data, args=(self.data_queue,))
        self.plotting_process.start()

    @staticmethod
    def _plot_data(data_queue):
        # Wait for data to be available
        while data_queue.empty():
            time.sleep(0.1)

        # Initialize data
        first_data = data_queue.get()['rewards'] # hardcoded for now
        data = {k: [v] for k, v in first_data.items()}

        # Create subplots
        fig, axs = plt.subplots(nrows=len(data)+4, ncols=1, figsize=(10, len(data) * 2)) # +4 = hack to allow for extra plots coming in later
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.99, top=0.97, wspace=0.4, hspace=0.4)

        # Update function for animation
        def update(frame):
            nonlocal data  # Declare data as nonlocal to modify it within this function

            # Retrieve new data from the queue
            while not data_queue.empty():
                new_data = data_queue.get()
                if new_data['done'] == True:
                    data = {k: [] for k in data}
                new_data = new_data['rewards']

                # Check for new keys and add them with prepended Nones
                new_keys = set(new_data.keys()) - set(data.keys())
                if new_keys:
                    for k in new_keys:
                        data[k] = [None for _ in list(data.values())[0]] + [new_data[k]]

                # Update stored data
                for k, v in new_data.items():
                    data[k].append(v)
                # Fill in missing data with Nones
                for k in set(data.keys()) - set(new_data.keys()):
                    data[k].append(None)

            # Update plots
            for i, (key, values) in enumerate(data.items()):
                axs[i].clear()
                axs[i].grid(True, which='both')
                axs[i].plot(values)
                axs[i].set_title(key)
                axs[i].set_xlim(0, len(values))

        ani = FuncAnimation(fig, update)
        plt.show()

    def __del__(self):
        self.plotting_process.terminate()

    def add_data(self, new_data, done):
        new_data['done'] = done
        self.data_queue.put(new_data)

