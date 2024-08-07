#include<iostream>

using namespace std;

int main(){

    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount( &count);

    for(int i=0; i<count; i++){
        cudaGetDeviceProperties(&prop, i);
        cout<<" ----------GEneral Information for device "<< i << " ---\n";
        cout<<"Name: " << prop.name<<'\n';

        cout<<"Compute capability: " << prop.major, prop.minor<<'\n';

        cout<<"Clock rate: " << prop.clockRate<<'\n';

        cout<<"Device copy overlap: \n" ;

        if(prop.deviceOverlap){
            cout<<"Enabled \n";
        }
        else{
                cout<<"Disabled \n";
            }

        cout<<"Kernal execution timeout: \n" ;

        if(prop.kernelExecTimeoutEnabled){
            cout<<"Enabled \n";
        }
        else{
                cout<<"Disabled \n";
            }


        cout<<" ----------Memory information for device "<< i << " -----\n";
        cout<<"Total global mem: " << prop.totalGlobalMem<<'\n';

        cout<<"Total constant mem: " << prop.totalConstMem<<'\n';

        cout<<"Max mem pitch: " << prop.memPitch<<'\n';

        cout<<"Texture Alignment: " << prop.textureAlignment<<'\n';

        ///////////////////////////////////////////

        cout<<" ----------MP information for device "<< i << " -----\n";
        cout<<"Multiprocessor count: " << prop.multiProcessorCount<<'\n';

        cout<<"shared mem per mp: " << prop.sharedMemPerBlock<<'\n';

        cout<<"Register per mp: " << prop.regsPerBlock<<'\n';

        cout<<"Threads in wrap: " << prop.warpSize<<'\n';



        cout<<"Max thread per block: " << prop.maxThreadsPerBlock<<'\n';

        cout<<"Max thread dimensions: " << prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]<<'\n';

        cout<<"Max grid dimensions: " << prop.maxGridSize[0],  prop.maxGridSize[1], prop.maxGridSize[2]<<'\n';

    }


    return 0;
}