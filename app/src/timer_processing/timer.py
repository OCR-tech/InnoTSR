# Import threading module to handle timer functionality
import threading


class Timer(object):
  """
  A Timer class that repeatedly executes a specified function at a given interval.
  """
  def __init__(self, interval, function, *args, **kwargs):
    """
    Initialize the Timer object.
    """
    self.interval = interval  # Time interval between executions
    self.function = function  # Function to execute
    self.args = args
    self.kwargs = kwargs
    self.timer = None
    self.is_running = False  # Flag to indicate if the timer is running
    self.start()  # Start the timer upon initialization

  def _run(self):
    """
    Internal method that executes the function and restarts the timer.
    """
    # print("//=== timer_run ===//")
    self.is_running = False  # Reset the running flag
    self.start()  # Restart the timer
    self.function(*self.args, **self.kwargs)  # Execute the function with arguments

  def start(self):
    """
    Start the timer if it is not already running.
    """
    # print("//=== timer_start ===//")
    if not self.is_running:  # Check if the timer is not already running
        self.timer = threading.Timer(self.interval, self._run)  # Create a new timer
        self.timer.start()  # Start the timer
        self.is_running = True  # Set the running flag to True

  def stop(self):
    """
    Stop the timer if it is running.
    """
    # print("//=== timer_stop ===//")
    if self.timer:  # Check if the timer exists
        self.timer.cancel()  # Cancel the timer
    self.is_running = False  # Set the running flag to False
