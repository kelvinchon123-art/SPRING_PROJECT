import time
import math
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button

WHEEL_RADIUS_CM = 2.5
WHEEL_CIRCUMFERENCE = 2 * math.pi * WHEEL_RADIUS_CM
TICKS_PER_REV = 20  

motor_left_speed = PWMOutputDevice(12, frequency=300)
motor_left_in1 = DigitalOutputDevice(5)
motor_left_in2 = DigitalOutputDevice(6)
    
motor_right_speed = PWMOutputDevice(19, frequency=300)
motor_right_in3 = DigitalOutputDevice(22)
motor_right_in4 = DigitalOutputDevice(26)

encoder_left = Button(5, pull_up=True)
encoder_right = Button(6, pull_up=True)

left_ticks = 0
right_ticks = 0

def count_left():
    global left_ticks
    left_ticks += 1

def count_right():
    global right_ticks
    right_ticks += 1

encoder_left.when_pressed = count_left
encoder_right.when_pressed = count_right

def set_motor(left_val, right_val):
    motor_left_in1.off()
    motor_left_in2.on()
    motor_left_speed.value = abs(left_val)
    
    motor_right_in3.off()
    motor_right_in4.on()
    motor_right_speed.value = abs(right_val)

def stop():
    motor_left_speed.value = 0
    motor_right_speed.value = 0

def test_duty_cycle(duty_cycle_decimal):
    global left_ticks, right_ticks
    
    left_ticks = 0
    right_ticks = 0
    
    # We now use your decimal directly as the speed
    speed = float(duty_cycle_decimal) 
    
    print(f"Testing Duty Cycle: {speed:.2f} (Frequency: 300Hz)")
    print("Running for exactly 1.00 second...")
    
    set_motor(speed, speed)
    time.sleep(1.0) 
    stop()
    
    # --- The Distance Math ---
    avg_ticks = (left_ticks + right_ticks) / 2.0
    revolutions = avg_ticks / float(TICKS_PER_REV)
    calculated_distance = revolutions * WHEEL_CIRCUMFERENCE
    
    print("🛑 Test Complete!")
    print(f"Left Ticks:          {float(left_ticks):.2f}")
    print(f"Right Ticks:         {float(right_ticks):.2f}")
    print(f"Average Ticks:       {avg_ticks:.2f}")
    print(f"Calculated Revs:     {revolutions:.2f} revolutions")
    print(f"Calculated Distance: {calculated_distance:.2f} cm")
    print("Please measure the Actual Distance and Angle Deviation now.")

# --- Run the Test Here ---
time.sleep(2)
test_duty_cycle(0.45) # Now inputs natively as a decimal! (e.g., 0.50, 0.75, 1.00)
