# TODO: implement a class for PID controller
class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: add aditional variables to store the current state of the controller
        self.integral = 0

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):
        print(sensor_readings)
        errors = [commanded_variable - sensor_readings[0], commanded_variable - sensor_readings[1]]
        self.integral += errors[0] * self.sensor_period

        derivative = (errors[0] - errors[1]) / self.sensor_period

        pid_output = (
            self.gain_prop * errors[0] +
            self.gain_int * self.integral +
            self.gain_der * derivative
        )

        return pid_output