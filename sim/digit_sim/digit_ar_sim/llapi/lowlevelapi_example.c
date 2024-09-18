/*
 * Copyright 2021 Agility Robotics
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "lowlevelapi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static double target_position[] = {
  -0.0462933,
  -0.0265814,
  0.19299,
  -0.3,
  -0.0235182,
  -0.0571617,
  0.0462933,
  0.0265814,
  -0.19299,
  0.3,
  -0.0235182,
  0.0571617,
  -0.3,
  0.943845,
  0.0,
  0.3633,
  0.3,
  -0.943845,
  0.0,
  -0.3633,
};


int main(int argc, char* argv[])
{
  // The publisher address should be changed to the ip address of the robot
  // const char* publisher_address = "127.0.0.1";
  const char* publisher_address = "10.10.1.1";
  llapi_init(publisher_address);

  // Define inputs and outputs (updated each iteration)
  llapi_command_t command = {0};
  llapi_observation_t observation;

  // Connect to robot (need to send commands until the subscriber connects)
  command.apply_command = false;
  while (!llapi_get_observation(&observation)) llapi_send_command(&command);

  // Get local copy of command limits (torque and damping)
  const llapi_limits_t* limits = llapi_get_limits();

  while (1) {
    // Update observation
    int return_val = llapi_get_observation(&observation);
    if (return_val < 1) {
      // Error occurred
    } else if (return_val) {
      // fprintf(stdout, "qpos: %f\n", observation.motor.position[0]);
    } else {
      // No new data
    }

    // // PD control on positions
    for (int i = 0; i < NUM_MOTORS; ++i) {
      command.motors[i].torque =
        150.0 * (target_position[i] - observation.motor.position[i]);
      command.motors[i].velocity = 0.0;
      command.motors[i].damping = 0.75 * limits->damping_limit[i];
    }
    command.fallback_opmode = Damping;
    command.apply_command = true;

    llapi_send_command(&command);
    printf("command sent\n");

    // Check if llapi has become disconnected
    if (!llapi_connected()) {
      // Handle error case. You don't need to re-initialize subscriber
      // Calling llapi_send_command will keep low level api open
    }

    // Sleep to keep to a reasonable update rate
    usleep(1000);
  }
}