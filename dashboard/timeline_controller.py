from nicegui import ui


class TimelineController:
    """ Plays back a trajectory by advancing the timeline slider at [tick_rate] frames per second """

    def __init__(self, tick_rate: int, max_ticks: int, play_button, timeline_slider):
        # store ui elements
        self.play_button = play_button
        self.timeline_slider = timeline_slider

        self.is_playing = False
        self.tick_rate = tick_rate
        self.current_tick = 0
        self.max_ticks = max_ticks
        self.timer = None

    def timer_callback(self):
        if self.is_playing:
            self.current_tick = min(self.current_tick + 1, self.max_ticks)
            self.timeline_slider.value = self.current_tick

            # Reached the end: rewind to the start and stop
            if self.current_tick >= self.max_ticks:
                self.current_tick = 0
                self.timeline_slider.value = self.current_tick
                self.pause()

    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self.play_button.text = "Pause"
            # Create timer with interval based on tick rate
            interval = 1.0 / self.tick_rate
            self.timer = ui.timer(interval, self.timer_callback)

    def pause(self):
        self.is_playing = False
        self.play_button.text = "Play"
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def toggle_playback(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def set_tick_rate(self, rate):
        try:
            self.tick_rate = max(0.1, float(rate))
            # Restart timer with new rate if playing
            if self.is_playing:
                self.pause()
                self.play()
        except (ValueError, TypeError):
            self.tick_rate = 1.0

    def slider_change(self, value):
        self.current_tick = value
