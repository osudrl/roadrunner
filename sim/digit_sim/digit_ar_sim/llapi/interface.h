#ifndef INTERFACE_H
#define INTERFACE_H

#pragma once

#include "lowlevelapi.h"
#include "udp.h"

#define COMMAND_PD_SIZE sizeof(llapi_command_pd_t)
#define OBSERVATION_SIZE sizeof(llapi_observation_t)
// sizeof(llapi_observation_t): 872
// sizeof(llapi_command_pd_t): 640

// Define the structure to hold the command data from python/policy
typedef struct {
    double kp[NUM_MOTORS];
    double kd[NUM_MOTORS];
    double position[NUM_MOTORS];
    double feedforward_torque[NUM_MOTORS];
} llapi_command_pd_t;

#ifdef __cplusplus
extern "C" {
#endif

// Main function to run the UDP server, initialize LLAPI, and run torque control loop
void llapi_run_udp(const char* robot_address_str);

// Helper functions to pack and unpack the command and observation data
void pack_command_pd(void* buffer, const llapi_command_pd_t* command);
void pack_observation(char* buffer, const llapi_observation_t* observation);
void unpack_command_pd(const char* buffer, llapi_command_pd_t* command);
void unpack_observation(const void* buffer, llapi_observation_t* observation);
static long long get_microseconds(void);

// Exit function to close the UDP socket
void auto_exit_function();

#ifdef __cplusplus
}
#endif

#endif // INTERFACE_H