#pragma OPENCL EXTENSION cl_intel_channels : enable

#define TILE_SIZE 64
#define WIDTH 640
#define HEIGHT 480
#define TILES_W (WIDTH / TILE_SIZE)
#define TILES_H (HEIGHT / TILE_SIZE)
#define PACKET_SIZE 3

typedef struct {
   uchar data[PACKET_SIZE][PACKET_SIZE];
} channel_packet;

channel channel_packet sobel_channel __attribute__((depth(32)));
channel uchar out_channel __attribute__((depth(32)));

__attribute__((max_global_work_dim(0)))
__kernel void mem_read(__global const uchar* restrict input, int width, int height) {
    

    __local uchar local_tile[TILE_SIZE+2][TILE_SIZE+2];
    
    for (int height_count = 0; height_count < TILES_H; height_count++) {
        for (int width_count = 0; width_count < TILES_W; width_count++) {

            for (int h = 0; h < TILE_SIZE+2; h++){
                for (int w = 0; w < TILE_SIZE+2; w++){

                    int global_h = height_count * TILE_SIZE + h - 1;
                    int global_w = width_count * TILE_SIZE + w - 1;
                    int mem_index = global_h * width + global_w;
                    
                    if (global_h < height && global_w < width) {
                        local_tile[h][w] = input[mem_index];
                    } else {
                        local_tile[h][w] = 0;
                    }

                }
            }

            for (int h = 0; h < TILE_SIZE; h++){
                for (int w = 0; w < TILE_SIZE; w++){
                    
                    channel_packet packet;
                    
                    #pragma unroll
                    for (int i = 0; i < PACKET_SIZE; i++){
                        #pragma unroll
                        for (int j = 0; j < PACKET_SIZE; j++){
                            
                            packet.data[i][j] = local_tile[h+i][w+j];

                        }
                    }

                    write_channel_intel(sobel_channel, packet);
                }
            }

            
            
        }
    }




}

__attribute__((max_global_work_dim(0)))
__kernel void sobel(
    int width,
    int height
) {

    const char sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const char sobel_y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};

    const int total_pixels_to_process = (TILES_H * TILE_SIZE) * (TILES_W * TILE_SIZE);

    for (int t = 0; t < total_pixels_to_process; t++) {

        channel_packet packet = read_channel_intel(sobel_channel);

        int gx = 0;
        int gy = 0;

        #pragma unroll
        for (int i = 0; i < PACKET_SIZE; i++){
            #pragma unroll
            for (int j = 0; j < PACKET_SIZE; j++){
                
                uchar pixel = packet.data[i][j];
                gx += pixel * sobel_x[i][j];
                gy += pixel * sobel_y[i][j];

            }
        }

        int magnitude = (int)sqrt((float)(gx * gx + gy * gy));
        if (magnitude > 255) magnitude = 255;
        write_channel_intel(out_channel, (uchar)magnitude);
        
    }

}

__attribute__((max_global_work_dim(0)))
__kernel void mem_write(
    __global uchar* restrict output,
    int width,
    int height
) {
    const int num_tiles_h = height / TILE_SIZE;
    const int num_tiles_w = width / TILE_SIZE;

    for (int height_count = 0; height_count < TILES_H; height_count++) {
        for (int width_count = 0; width_count < TILES_W; width_count++) {

            for (int h = 0; h < TILE_SIZE; h++){
                for (int w = 0; w < TILE_SIZE; w++){
                    
                    uchar pixel_out = read_channel_intel(out_channel);
                    
                    int global_h = height_count * TILE_SIZE + h;
                    int global_w = width_count * TILE_SIZE + w;

                    if (global_h < height && global_w < width) {
                        output[global_h * width + global_w] = pixel_out;
                    }
                }
            }
        }
    }
}