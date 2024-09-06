#include "interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <time.h>
#include <signal.h>


void auto_exit_function() {
    printf("Exiting the program and closing UDP socket.\n");
    llapi_free();
    exit(EXIT_SUCCESS);
}


static long long get_microseconds(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000000 + now.tv_nsec / 1000;
}


void unpack_observation(const void* buffer, llapi_observation_t* observation)
{
    // Copy the buffer into the observation struct
    memcpy(observation, buffer, sizeof(llapi_observation_t));
}


void unpack_command_pd(const char* buffer,
                       llapi_command_pd_t* command)
{
    // Copy the buffer into the command struct
    memcpy(command, buffer, sizeof(llapi_command_pd_t));
}


void pack_command_pd(void* buffer, const llapi_command_pd_t* command)
{
    if (command == NULL || buffer == NULL) {
        // Handle the error, e.g., return or raise an exception
        return;
    }

    // // Copy the command struct into the buffer
    memcpy(buffer, command, sizeof(llapi_command_pd_t));
}


void pack_observation(char *buffer, const llapi_observation_t *observation)
{
    if (observation == NULL || buffer == NULL) {
        // Handle the error, e.g., return or raise an exception
        return;
    }

    // Copy the command struct into the buffer
    memcpy(buffer, observation, sizeof(llapi_observation_t));
}


void llapi_run_udp(const char *robot_address_str) {
    signal(SIGINT, auto_exit_function);

    // Loop frequency and timeout
    // NOTE: Running loop frequency faster than 2kHz since UDP.
    const long long cycle_usec = 1000000 / 5000;
    const long long timeout_usec = 100000;

    // Option for UDP socket (Python/C)
    const char *remote_addr_str = "127.0.0.1"; // address to send commands to
    const char *remote_port_str = "35001";
    const char *iface_addr_str = "0.0.0.0"; // local address to initialize socket
    const char *iface_port_str = "35000";

    // Bind to UDP socket
    // int sock = udp_init_client(remote_addr_str, remote_port_str, iface_addr_str, iface_port_str);
    int sock = udp_init_host(iface_addr_str, iface_port_str);
    if (-1 == sock) {
        printf("Error: Could not bind to UDP socket at %s:%s\n", iface_addr_str, iface_port_str);
        exit(EXIT_FAILURE);
    }

    // Create I/O size and allocate memory
    int dinlen, doutlen;
    dinlen = sizeof(llapi_command_pd_t);
    doutlen = sizeof(llapi_observation_t);

    const int recvlen = PACKET_HEADER_LEN + dinlen;
    const int sendlen = PACKET_HEADER_LEN + doutlen;
    unsigned char *recvbuf = (unsigned char *)malloc(recvlen);
    unsigned char *sendbuf = (unsigned char *)malloc(sendlen);

    // Seperate I/O buffers into header and data
    const unsigned char *recvheader = recvbuf;
    const unsigned char *recvdata = &recvbuf[PACKET_HEADER_LEN];
    unsigned char *sendheader = sendbuf;
    unsigned char *senddata = &sendbuf[PACKET_HEADER_LEN];

    // Create observation and command structs
    llapi_observation_t observation;
    llapi_command_pd_t command_pd = {0};
    llapi_command_t command = {0};

    // Create header information struct
    packet_header_info_t header_info = {0};

    // Setup remote address and port
    // Convert the IP address from the string representation to binary format
    struct sockaddr_in dst_addr;
    memset(&dst_addr, 0, sizeof(dst_addr));
    dst_addr.sin_family = AF_INET;
    if (inet_pton(AF_INET, remote_addr_str, &dst_addr.sin_addr) != 1) {
        printf("Invalid IP address format: %s\n", remote_addr_str);
        exit(EXIT_FAILURE);
    }
    // Assign remote port
    int remote_port = atoi(remote_port_str);
    dst_addr.sin_port = htons(remote_port);

    // Prepare initial null command packet to start communication
    printf("Connecting to Digit...\n");
    memset(sendbuf, 0, sendlen);
    bool received_data = false;

    // Initialize LLAPI
    // NOTE: This is still a UDP socket, but the socket cannot be explicitly accessed.
    llapi_init(robot_address_str);

    // Connect to robot (need to send commands until the subscriber connects)
    command.apply_command = false;
    while (!llapi_get_observation(&observation)) {
        llapi_send_command(&command);
    }
    printf("Connected with Digit LLAPI at %s\n", robot_address_str);

    // Main loop to receive and send data between Python/C and LLAPI/Robot
    printf("Starting main torque control loop with LLAPI...\n");
    long long time_loop_start = get_microseconds();
    while (true) {
        // Get policy/PD command from python side
        // Get newest packet, or return -1 if no new packets are available
        ssize_t nbytes;
        nbytes = get_newest_packet(sock, recvbuf, recvlen, NULL, 0);
        if (nbytes == recvlen) {
            // Process incoming header and write outgoing header
            process_packet_header(&header_info, recvheader, sendheader);
            unpack_command_pd(recvdata, &command_pd);
            // printf("time for torque control at %f with policy first action %f\n", observation.time, command_pd.position[0]);
        }
        else {}

        // Get observation from robot
        int return_code = llapi_get_observation(&observation);
        // TODO: Handle cases where observation is not updated

        // Construct torque command
        for (int i = 0; i < NUM_MOTORS; ++i) {
            command.motors[i].torque =
                command_pd.kp[i] * (command_pd.position[i] - observation.motor.position[i]) +
                command_pd.feedforward_torque[i];
            command.motors[i].velocity = 0.0;
            command.motors[i].damping = command_pd.kd[i];
        }
        command.apply_command = true;
        command.fallback_opmode = Locomotion;
        llapi_send_command(&command);

        // Check if llapi has become disconnected
        if (!llapi_connected()) {
            // Handle error case. You don't need to re-initialize subscriber
            // Calling llapi_send_command will keep low level api open
            // TODO: Make a fallback function to detect when the obs or PD command not updated
            // for some time so auto swtich to Locomotion mode as a fallback.
            // This will be the fallback mode ontop of Agility's controller.
        }

        // Update observation from robot and send to python side
        llapi_get_observation(&observation);
        pack_observation(senddata, &observation);
        send_packet(sock, sendbuf, sendlen, (struct sockaddr *)&dst_addr, sizeof(dst_addr));
        while (get_microseconds() - time_loop_start < cycle_usec) {}
        time_loop_start = get_microseconds();
    }
}


int main(int argc, char* argv[])
{
    signal(SIGINT, auto_exit_function);
    const char *robot_address_str = "127.0.0.1";
    llapi_run_udp(robot_address_str);
    return 0;
}
