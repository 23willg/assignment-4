__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth, int imageChannels)
{
    //@@ Insert code to implement matrix multiplication here
    //height or row
    int i = get_global_id(0); 
    // printf("i=%d\n", i);
    //width or col
    int j = get_global_id(1);
    // printf("j=%d\n", j);

    int maskRadius = (maskWidth / 2);

    for (int k = 0; k < imageChannels; k++)
    {
        float accum = 0;
        for (int y = -maskRadius; y <= maskRadius; y++)
        {
            for (int x = -maskRadius; x <= maskRadius; x++)
            {

                int xOffset = j + x;
                int yOffset = i + y;
                // printf("xOff%d\n", xOffset);
                // printf("yOff%d\n", yOffset);

                if (((xOffset >= 0) && (xOffset < width)) && ((yOffset >= 0) && (yOffset < height)))
                {
                    float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                    float maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];

                    accum += imagePixel * maskValue;
                }
            }
        }

        if (accum < 0)
        {
            accum = 0;
        }
        else if (accum > 1)
        {
            accum = 1;
        }
        
        outputData[(i * width + j)*imageChannels + k] = accum;
    }
}