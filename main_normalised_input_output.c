// Jim C. Williams (github.com/jcwml)
// gcc main.c -lm -Ofast -mavx -mfma -o main
// I have used the TFCNNv3 library (github.com/tfcnn)
// by James William Fletcher to teach a neural network
// to learn a fixed frequency cutoff filter

#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>

#include "TFCNNv3.h"

#include <locale.h>
#include <sys/time.h>
#include <sys/file.h>
#include <sys/mman.h>

#define DEBUG 0
uint DSS = 0;       // dataset size (num of 1-byte unsigned)
uint SONGLEN = 0;   // song size also in bytes
unsigned char* train_x;
unsigned char* train_y;
unsigned char* song;
network net;
uint EPOCHS = 333333333;
time_t st;

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

void timestamp(char* ts)
{
    const time_t tt = time(0);
    strftime(ts, 16, "%H:%M:%S", localtime(&tt));
}

unsigned char quantise_float(float f)
{
    if(f < 0.f){f = 0.f;}else{f += 0.5f;}
    return (unsigned char)f;
}

void generate_output(int sig_num) 
{
    if(sig_num == 2){printf(" Early termination called.\n\n");}

    char strts[16];
    timestamp(&strts[0]);
    printf("\n[%s] Training Ended.\nTime Taken: %lu seconds (%.2f minutes).\n", strts, time(0)-st, ((f32)(time(0)-st))/60.f);

    // generate output raw audio
    timestamp(&strts[0]);
    printf("\n[%s] Generating output audio.\n", strts);
    st = time(0);
    int f = creat("song_output.raw", O_WRONLY | S_IRWXU);
    if(f > -1)
    {
        const int SONGLEN4 = SONGLEN-4;
        for(uint i=4; i < SONGLEN4; i++)
        {
            const int ofs = i-4;
            f32 input[9];
            f32 output = 0.f;
            for(int j = 0; j < 9; j++)
                input[j] = (((f32)song[ofs+j])-128.f)*0.0078125f;

            const f32 loss = processNetwork(&net, &input[0], NULL, &output);

            const unsigned char outputq = quantise_float((output+1.f)*128.f);
            //printf("[%u] %f %u\n", i, output, outputq);
            if(write(f, &outputq, sizeof(unsigned char)) != sizeof(unsigned char))
            {
                printf("write error\n");
                close(f);
                exit(3);
            }
        }

        close(f);
    }
    else
    {
        printf("Could not open song_output.raw for writing.\n");
        exit(3);
    }
    timestamp(&strts[0]);
    printf("[%s] Audio Generation Ended.\nTime Taken: %lu seconds (%.2f minutes).\n", strts, time(0)-st, ((f32)(time(0)-st))/60.f);
    
    // save network
    saveNetwork(&net, "network.save");

    // done
    destroyNetwork(&net);
    munmap(train_x, DSS);
    munmap(train_y, DSS);
    munmap(song, SONGLEN);
    exit(0);
}

int main()
{
    // ctrl+c callback
    signal(SIGINT, generate_output);

    // setlocale
    setlocale(LC_NUMERIC, "");

    // log start time
    char strts[16];
    timestamp(&strts[0]);
    printf("\n[%s] Dataset Loading Started.\n", strts);
    st = time(0);

    // load dataset
    int f = open("train_x.raw", O_RDONLY);
    if(f > -1)
    {
        const size_t len = lseek(f, 0, SEEK_END);
        DSS = len;
        train_x = mmap(NULL, len, PROT_READ, MAP_SHARED, f, 0);
        if(train_x == MAP_FAILED)
        {
            printf("mmap() on train_x.raw failed.\n");
            close(f);
            return 1;
        }
        close(f);
    }
    else
    {
        printf("train_x.raw not found.\n");
        return 1;
    }
    f = open("train_y.raw", O_RDONLY);
    if(f > -1)
    {
        const size_t len = lseek(f, 0, SEEK_END);
        if(DSS != len)
        {
            printf("train_x.raw and train_y.raw are not the same length / file size.\n");
            close(f);
            return 1; 
        }
        train_y = mmap(NULL, len, PROT_READ, MAP_SHARED, f, 0);
        if(train_y == MAP_FAILED)
        {
            printf("mmap() on train_y.raw failed.\n");
            close(f);
            return 1;
        }
        close(f);
    }
    else
    {
        printf("train_y.raw not found.\n");
        return 1;
    }
    f = open("song.raw", O_RDONLY);
    if(f > -1)
    {
        const size_t len = lseek(f, 0, SEEK_END);
        SONGLEN = len;
        song = mmap(NULL, len, PROT_READ, MAP_SHARED, f, 0);
        if(song == MAP_FAILED)
        {
            printf("mmap() on song.raw failed.\n");
            close(f);
            return 1;
        }
        close(f);
    }
    else
    {
        printf("song.raw not found.\n");
        return 1;
    }

    timestamp(&strts[0]);
    printf("[%s] Dataset Loading Ended.\nTime Taken: %lu seconds (%.2f minutes).\n", strts, time(0)-st, ((f32)(time(0)-st))/60.f);
    printf("[%s] Training Started.\n\n", strts);
    st = time(0);
    
    // init network
    int r = createNetwork(&net, WEIGHT_INIT_NORMAL_GLOROT, 9, 1, 3, 32, 1); 
    if(r < 0){printf("Init network failed, error: %i\n", r); return 2;}

    // config network
    // setWeightInit(&net, WEIGHT_INIT_NORMAL_GLOROT);
    // setGain(&net, 1.f);
    // setUnitDropout(&net, 0.f);
    setLearningRate(&net, 0.001f);
    setActivator(&net, TANH);
    setOptimiser(&net, OPTIM_ADAGRAD);
    setBatches(&net, 1);

    // train network
    uint epochs_per_second = 0;
    uint epoch_seconds = 0;
    for(uint j=0; j < EPOCHS; j++)
    {
        f32 epoch_loss = 0.f;
        const uint DSS4 = DSS-4;
        for(uint i=4; i < DSS4; i++)
        {
            const int ofs = i-4;
            f32 txf[9];
            for(int j = 0; j < 9; j++)
                txf[j] = (((f32)train_x[ofs+j])-128.f)*0.0078125f;
            const f32 tyf = (((f32)train_y[ofs])-128.f)*0.0078125f;

            const f32 loss = processNetwork(&net, &txf[0], &tyf, NULL);
            epoch_loss += loss;

            static time_t lt = 0;
            if(time(0) > lt)
            {
                static uint ls = 0;
#if DEBUG == 1
                layerStat(&net);
#endif
                printf("[%'u/%'u] < [%'u]\n", i, DSS, DSS-i);
                printf("[%u] loss: %g\n", i, loss);
                printf("[%u] avg loss: %g\n", i, epoch_loss/(i-4));
                printf("[%u] delta-iter: %'u\n\n", i, i-ls);
                lt = time(0)+3;
                ls = i;
            }

            // !!! YOU MAY WANT TO COMMENT OUT THIS LINE
            //if(i > 600333 && epoch_loss/i < 0.5f){generate_output(0);}
        }

        printf("[%u] epoch loss: %g\n", j, epoch_loss);
        printf("[%u] avg epoch loss: %g\n\n", j, epoch_loss/(DSS4-4));
    }

    // training done let's use the trained network to produce an output
    generate_output(0);
    return 0;
}
