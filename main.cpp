#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mach/mach_time.h>
#include <stdio.h>      /* printf */
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <math.h>       /* sqrt */
#include <sys/time.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
 #include "vectorclass.h"
#include <immintrin.h>
#include "FileReader.h"
#include "column.h"
#include "table.h"
#include "perftimers.h"

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300)

#include <immintrin.h>

int check_4th_gen_intel_core_features()
{
    const int the_4th_gen_features =
        (_FEATURE_AVX2 | _FEATURE_FMA | _FEATURE_BMI | _FEATURE_LZCNT | _FEATURE_MOVBE);
    return _may_i_use_cpu_feature( the_4th_gen_features );
}

#else /* non-Intel compiler */

#include <stdint.h>
#if defined(_MSC_VER)
# include <intrin.h>
#endif

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
#if defined(_MSC_VER)
    __cpuidex(abcd, eax, ecx);
#else
    uint32_t ebx, edx;
# if defined( __i386__ ) && defined ( __PIC__ )
     /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
# else
    __asm__ ( "cpuid" : "+b" (ebx),
# endif
              "+a" (eax), "+c" (ecx), "=d" (edx) );
    abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
#endif
}

int check_xcr0_ymm()
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);  /* min VS2010 SP1 compiler is required */
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif
    return ((xcr0 & 6) == 6); /* checking if xmm and ymm state are enabled in XCR0 */
}


int check_4th_gen_intel_core_features()
{
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1   &&
       CPUID.(EAX=01H, ECX=0H):ECX.MOVBE[bit 22]==1 &&
       CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 */
    run_cpuid( 1, 0, abcd );
    if ( (abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask )
        return 0;

    if ( ! check_xcr0_ymm() )
        return 0;

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask )
        return 0;

    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    run_cpuid( 0x80000001, 0, abcd );
    if ( (abcd[2] & (1 << 5)) == 0)
        return 0;

    return 1;
}

#endif /* non-Intel compiler */


static int can_use_intel_core_4th_gen_features()
{
    static int the_4th_gen_features_available = -1;
    /* test is performed once */
    if (the_4th_gen_features_available < 0 )
        the_4th_gen_features_available = check_4th_gen_intel_core_features();

    return the_4th_gen_features_available;
}

// Test work space to understand the different SIMD operators
void simdExperiments()
{
    int resutlsArray[8];
    bool bresutlsArray[4];

    Vec8i a(10, 11, 12, 13,10, 11, 12, 13);
    Vec8i b(14, 13, 12, 11,14, 13, 12, 11);
    Vec8i c = a > b;

    Vec4i ai(10, 11, 12, 13);
    Vec4i bi(14, 13, 12, 11);
    Vec4ib ci = ai <= bi;
    ci.store(bresutlsArray);
    bool bb = true;
    for (int i=0;i<4;i++)
    {
        cout << ci[i]<<" "<< bresutlsArray[i]<<endl;
    }


    Vec8ib d = (Vec8ib)a > b;
    c.store(resutlsArray);

    Vec8i zeroVector(0,0,0,0,0,0,0,0);
    Vec8i oneVector(1, 1, 1, 1, 1, 1, 1, 1);

    Vec8i result = oneVector++;
    result.store(resutlsArray);
    result++;
    result.store(resutlsArray);
}

//Raw mach_absolute_times going in, difference in seconds out
double subtractTimes( uint64_t endTime, uint64_t startTime )
{
    uint64_t difference = endTime - startTime;
    static double conversion = 0.0;
    
    if( conversion == 0.0 )
    {
        mach_timebase_info_data_t info;
        kern_return_t err = mach_timebase_info( &info );
        
		//Convert the timebase into seconds
        if( err == 0  )
			conversion = 1e-9 * (double) info.numer / (double) info.denom;
    }
    
    return conversion * (double) difference;
}

int GetSumScalar(int* intArray,int arrayLength)
{
    int sum = 0;
    for (int i = 0; i < arrayLength; i++)
    {
     sum += intArray[i];
    }
    return sum;
}


// Implements (A* x B*)
int* MultTwoArraysScalar(int* vector1, int* vector2,int arrayLength)
{
    Performancetimer* pt = new Performancetimer("MultTwoArraysScalar");

    int* resultsArray = new int[arrayLength];
    for (int i = 0 ; i < arrayLength;i++)
    {
        resultsArray[i] = vector1[i] * vector2[i];
    }
    pt->GetReport();
    return resultsArray;
}

// Implements (A* x B*)
int* MultTwoArraysSIMD(int* vector1, int* vector2,int arrayLength)
{
    Performancetimer* pt = new Performancetimer("MultTwoArraysSIMD");
    int i = 0;
    int resultsArray[arrayLength];
    Vec8i tempSumVector, tempVector1,tempVector2;
    //Vec4i
    const int vectorsize = 8;
    // (AND-ing with -vectorsize will round down to nearest
    // lower multiple of vectorsize. This works only if
    // vectorsize is a power of 2)
    const int regularpart = arrayLength & (-vectorsize); // = 128

    // loop for 8 numbers at a time
    for ( i = 0; i < regularpart; i += vectorsize)
    {
         tempVector1.load(vector1+i); // load 8 elements
         tempVector2.load(vector2+i); // load 8 elements
         tempSumVector = tempVector1 * tempVector2; // add 8 elements
         tempSumVector.store_a(resultsArray+i);
    }

    for (; i < arrayLength; i++) {
     resultsArray[i] = vector1[i] * vector2[i];
    }

    pt->GetReport();
    return resultsArray;
}

// Implements Sum(A* x B*)
int SumAddTwoArraysScalar(int* vector1, int* vector2,int arrayLength)
{
    Performancetimer* pt = new Performancetimer("SumMultTwoArraysScalar");
    int64_t sum = 0;
    for (int i = 0 ; i < arrayLength;i++)
    {
        sum += vector1[i] + vector2[i];
    }
    pt->GetReport();
    return sum;
}

// Implements Sum(A* x B*)
float SumMultTwoArraysScalarfloat(float* vector1, float* vector2,int arrayLength,int loopCount)
{
    Performancetimer* pt = new Performancetimer("SumMultTwoArraysScalarfloat");
    float sum = 0;
    for (int i=0; i < loopCount; i++)
    {

        sum = 0;
        for (int i = 0 ; i < arrayLength;i++)
        {
            sum +=  vector1[i]*(1-vector2[i]);
        }
    }
    pt->GetReport();
    return sum;
}

// Implements Sum(A* + B*) in SIMD
int SumAddTwoArraysSIMD(int* vector1, int* vector2,int arrayLength)
{
    Performancetimer* pt = new Performancetimer("SumMultTwoArraysSIMD");
    int i = 0;
    int64_t sum = 0;
    Vec8i tempSumVector, tempVector1,tempVector2;
    const int vectorsize = 8;
    // (AND-ing with -vectorsize will round down to nearest
    // lower multiple of vectorsize. This works only if
    // vectorsize is a power of 2)
    const int regularpart = arrayLength & (-vectorsize); // = 128
    float initVector [vectorsize] =  { 0, 0, 0, 0, 0, 0, 0, 0 };
    tempSumVector.load(initVector);

    // loop for 8 numbers at a time
    for ( i = 0; i < regularpart; i += vectorsize)
    {
         tempVector1.load(vector1+i); // load 8 elements
         tempVector2.load(vector2+i); // load 8 elements
         tempSumVector += tempVector1 + tempVector2; // add 8 elements
    }

    for (; i < arrayLength; i++) {
     sum += vector1[i] + vector2[i];
    }

    sum += horizontal_add(tempSumVector); // add the vector sum

    pt->GetReport();
    return sum;
}

float simdFoatTest()
{
    float value = 1.0;
    float sum = 0;
    Vec8f tempSumVector, tempVector1,tempVector2,tempVector3;
    float scalarValue1 [] = { 21168.23,45983.16,13309.6,28955.64,22824.48,49620.16,44694.46,54058.05};
    float scalarValue2 [] = { value, value, value, value, value,value,value,value };
    float scalarValue3 [] = { 0.04,0.09,0.1,0.09,0.1,0.07,0,0.06};
    tempVector1.load(scalarValue1);
    tempVector2.load(scalarValue2);
    tempVector3.load(scalarValue3);

    tempSumVector = tempVector1 * (tempVector2 -  tempVector3); // add 8 elements

    sum += horizontal_add(tempSumVector); // add the vector sum
    return sum;
}

void simdMultTest(int* orderkeyArray,int dataSize)
{
    int* resultsArrayScalar = MultTwoArraysScalar(orderkeyArray,orderkeyArray,dataSize);
    int* resultsArraySIMD = MultTwoArraysSIMD(orderkeyArray,orderkeyArray,dataSize);
}

// Implements Sum(A* x (1-B*)) in SIMD
float SumMultTwoArraysSIMDfloat(float* vector1, float* vector2,int arrayLength,int loopCount)
{
    Performancetimer* pt = new Performancetimer("SumMultTwoArraysSIMDfloat");
    float sum = 0;
    for (int j=0; j < loopCount; j++)
    {
    int i = 0;
    sum = 0;
    Vec8f tempSumVector(0), tempVector1,tempVector2,tempVector3;

    const int vectorsize = 8;
    float initVector [vectorsize] =  { 0, 0, 0, 0, 0, 0, 0, 0 };
    float scalarValue [vectorsize] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    tempVector3.load(scalarValue);
    tempSumVector.load(initVector);

    // (AND-ing with -vectorsize will round down to nearest
    // lower multiple of vectorsize. This works only if
    // vectorsize is a power of 2)
    const int regularpart = arrayLength & (-vectorsize); // = 128

    // loop for 8 numbers at a time
    for ( i = 0; i < regularpart; i += vectorsize)
    {
         tempVector1.load(vector1+i); // load 8 elements
         tempVector2.load(vector2+i); // load 8 elements
         tempSumVector +=  tempVector1*(tempVector3 - tempVector2); // add 8 elements
    }

    for (; i < arrayLength; i++) {
     sum += vector1[i]*(1- vector2[i]);
    }

    sum += horizontal_add(tempSumVector); // add the vector sum
    }
    pt->GetReport();
    return sum;
}


// Implements Sum(A* x (1-B*)*(1+C) in Scalar
//(l_extendedprice*(1-l_discount)*(1+l_tax))
float Q1Expression2Scalar(float* vector1, float* vector2,float* vector3,int arrayLength,int loopCount)
{
    Performancetimer* pt = new Performancetimer("Q1Expression2Scalar");
    float sum = 0;
    for (int i=0; i < loopCount; i++)
    {

        sum = 0;
        for (int i = 0 ; i < arrayLength;i++)
        {
            sum +=  vector1[i]*(1-vector2[i])*(1+vector3[i]);
        }
    }
    pt->GetReport();
    return sum;
}

// Implements Sum(A* x (1-B*)*(1+C) in SIMD
//(l_extendedprice*(1-l_discount)*(1+l_tax))
float Q1Expression2SIMD(float* vector1, float* vector2,float* vector3,int arrayLength,int loopCount)
{
    Performancetimer* pt = new Performancetimer("Q1Expression2SIMD");
    float sum = 0;
    for (int j=0; j < loopCount; j++)
    {
        int i = 0;
        sum = 0;
        Vec8f tempSumVector(0), tempVector1,tempVector2,tempVector3(1),tempVector4;

        const int vectorsize = 8;
        float initVector [vectorsize] =  { 0, 0, 0, 0, 0, 0, 0, 0 };
        tempSumVector.load(initVector);

        float scalarValue [vectorsize] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        tempVector3.load(scalarValue);

        // (AND-ing with -vectorsize will round down to nearest
        // lower multiple of vectorsize. This works only if
        // vectorsize is a power of 2)
        const int regularpart = arrayLength & (-vectorsize); // = 128

        // loop for 8 numbers at a time
        for ( i = 0; i < regularpart; i += vectorsize)
        {
             tempVector1.load(vector1+i); // load 8 elements
             tempVector2.load(vector2+i); // load 8 elements
             tempVector4.load(vector3+i); // load 8 elements
             tempSumVector +=  tempVector1*(tempVector3 - tempVector2)*(tempVector3 + tempVector4);
        }

        for (; i < arrayLength; i++) {
         sum += vector1[i]*(1- vector2[i])*(1+vector3[i]);
        }

        sum += horizontal_add(tempSumVector); // add the vector sum
    }
    pt->GetReport();
    return sum;
}

int GetSumSIMD(int* intArray,int arrayLength)
{
    int sum = 0;
    int i = 0;
    Vec8i sum1(0), temp;
    //Vec4i
    const int vectorsize = 8;
    // (AND-ing with -vectorsize will round down to nearest
    // lower multiple of vectorsize. This works only if
    // vectorsize is a power of 2)
    const int regularpart = arrayLength & (-vectorsize); // = 128

    // loop for 8 numbers at a time
    for ( i = 0; i < regularpart; i += vectorsize) {
     temp.load(intArray+i); // load 8 elements
     sum1 += temp; // add 8 elements
    }

    for (; i < arrayLength; i++) {
     sum += intArray[i];
    }
    sum += horizontal_add(sum1); // add the vector sum

    return sum;
}

void InitArray(int* mydata, int arrayLength)
{
    for( int i = 0; i < arrayLength; i++)
    {
        mydata[i]=1;
    }
}

void SIMDSumArray(int* mydata,int loopCount,int datasize)
{
    int sum=0;
    Performancetimer* pt = new Performancetimer("SIMDSumArray");

    for (int i=0; i < loopCount; i++)
    {
        sum = GetSumSIMD(mydata,datasize);
    }
    printf("SIMD Sum %ld \n",sum);

    pt->GetReport();
}

void ScalarSumArray(int* mydata,int loopCount,int datasize)
{
    int sum;
    Performancetimer* pt = new Performancetimer("ScalarSumArray");

    for (int i=0; i < loopCount; i++)
    {
        sum = GetSumScalar(mydata,datasize);
    }

    pt->GetReport();

    printf("Scalar Sum %d \n",sum);
}

void CheckCpuFeatures()
{
    if ( can_use_intel_core_4th_gen_features() )
            printf("This CPU supports ISA extensions introduced in Haswell\n");
        else
            printf("This CPU does not support all ISA extensions introduced in Haswell\n");
}


struct Q1Results
{
    float sum_qty;          //sum(l_quantity) as sum_qty,
    float sum_base_price;   //sum(l_extendedprice) as sum_base_price,
    float sum_disc_price;   //sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
    float sum_charge;       //sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
    float avg_qty;          //avg(l_quantity) as avg_qty,
    float avg_price;        //avg(l_extendedprice) as avg_price,
    float avg_disc;         //avg(l_discount) as avg_disc,
    float count_order;      //count(*) as count_order
};

// Implements Q1 without Group by or filter 
Q1Results RunQ1Scalar (float* l_quantity, float* l_extendedprice,
                       float* l_discount, float* l_tax, int arrayLength,
                       int loopCount)
{
    Performancetimer* pt = new Performancetimer("RunQ1Scalar");
    Q1Results scalarQ1;

    for (int i=0; i < loopCount; i++)
    {
        float sum_qty = 0;          //sum(l_quantity) as sum_qty,
        float sum_base_price = 0;   //sum(l_extendedprice) as sum_base_price,
        float sum_disc_price = 0;   //sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
        float sum_charge = 0;       //sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
        float avg_qty = 0;          //avg(l_quantity) as avg_qty,
        float avg_price = 0;        //avg(l_extendedprice) as avg_price,
        float avg_disc = 0;         //avg(l_discount) as avg_disc,
        float count_order = 0;      //count(*) as count_order
        float exp1 = 0;             //cache the common expression
        float exp2 = 0;             //cache the common expression

        for (int i = 0 ; i < arrayLength;i++)
        {
            sum_qty         +=l_quantity[i];
            sum_base_price  +=l_extendedprice[i];
            avg_disc        +=l_discount[i]; // willl be used for average later on

            // no need to caluclate l_extendedprice*(1-l_discount) twice
            exp1            = l_extendedprice[i] * (1-l_discount[i]);
            exp2            = exp1*(1+l_tax[i]);

            sum_disc_price  +=exp1;
            sum_charge      +=exp2;
        }
        scalarQ1.sum_qty            = sum_qty;
        scalarQ1.sum_base_price     = sum_base_price;
        scalarQ1.sum_disc_price     = sum_disc_price;
        scalarQ1.sum_charge         = sum_charge;
        scalarQ1.avg_qty            = sum_qty / arrayLength;
        scalarQ1.avg_price          = sum_base_price / arrayLength;
        scalarQ1.avg_disc           = avg_disc / arrayLength;
        scalarQ1.count_order        = arrayLength;
    }
    pt->GetReport();
    return scalarQ1;
}







// Implements Q1 without Group by or filter using SIMD
Q1Results RunQ1SIMD(float* l_quantity, float* l_extendedprice,
                  float* l_discount, float* l_tax, int arrayLength,
                  int loopCount)
//(float* vector1, float* vector2,float* vector3,int arrayLength,int loopCount)
{
    Performancetimer* pt = new Performancetimer("RunQ1SIMD");
    float sum = 0;
    Q1Results q1results;

    for (int j=0; j < loopCount; j++)
    {
        int i = 0;
        sum = 0;
        const int vectorsize = 8;
        float initVector [vectorsize] =  { 0, 0, 0, 0, 0, 0, 0, 0 };
        float scalarValue [vectorsize] = { 1, 1, 1, 1, 1, 1, 1, 1 };

        // The vectors used for input columns
        Vec8f l_quantityVector, l_extendedpriceVector,l_discountVector,l_taxVector, scalarOneVector(1);

        Vec8f sum_qty(0);          //sum(l_quantity) as sum_qty,
        Vec8f sum_base_price(0);   //sum(l_extendedprice) as sum_base_price,
        Vec8f sum_disc_price(0);   //sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
        Vec8f sum_charge(0);       //sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
        Vec8f avg_disc(0);         //avg(l_discount) as avg_disc,
        Vec8f exp1(0);             //cache the common expression
        Vec8f exp2(0);             //cache the common expression

        // initialize the vectors
        sum_qty.load(initVector);
        sum_base_price.load(initVector);
        sum_disc_price.load(initVector);
        sum_charge.load(initVector);
        avg_disc.load(initVector);
        exp1.load(initVector);
        exp2.load(initVector);
        scalarOneVector.load(scalarValue);

        // (AND-ing with -vectorsize will round down to nearest
        // lower multiple of vectorsize. This works only if
        // vectorsize is a power of 2)
        const int regularpart = arrayLength & (-vectorsize); // = 128

        // loop for 8 numbers at a time
        for ( i = 0; i < regularpart; i += vectorsize)
        {
             l_quantityVector.load(l_quantity+i); // load 8 elements
             l_extendedpriceVector.load(l_extendedprice+i); // load 8 elements
             l_discountVector.load(l_discount+i); // load 8 elements
             l_taxVector.load(l_tax+i); // load 8 elements

             sum_qty            +=l_quantityVector;
             sum_base_price     +=l_extendedpriceVector;
             avg_disc           +=l_discountVector; // willl be used for average later on

             // no need to caluclate l_extendedprice*(1-l_discount) twice
             exp1            = l_extendedpriceVector * (scalarOneVector-l_discountVector);
             exp2            = exp1*(scalarOneVector+l_taxVector);

             sum_disc_price  +=exp1;
             sum_charge      +=exp2;
        }

        q1results.sum_qty           = horizontal_add(sum_qty);
        q1results.sum_base_price    = horizontal_add(sum_base_price);
        q1results.sum_disc_price    = horizontal_add(sum_disc_price);
        q1results.sum_charge        = horizontal_add(sum_charge);
        q1results.avg_qty           = horizontal_add(sum_qty) / arrayLength;
        q1results.avg_price         = horizontal_add(sum_base_price) / arrayLength;
        q1results.avg_disc          = horizontal_add(avg_disc) / arrayLength;
        q1results.count_order       = arrayLength;

        /*
        for (; i < arrayLength; i++) {
         sum += vector1[i]*(1- vector2[i])*(1+vector3[i]);
        }
       sum += horizontal_add(tempSumVector); // add the vector sum
       */
    }
    pt->GetReport();
    return q1results;
}


int main() {

    int datasize = 1000000;
    const int readRowCount = 1000000;
    datasize = readRowCount;
    int loopCount = 1000;

    // Check if SIMD is supported
    CheckCpuFeatures();

    // File parameters
    string flatFileName = "/Users/mmokhtar/Downloads/tpch_2_16_1/dbgen/1Gb/lineitem2.tbl.csv";
    string columnDelimeter = "|";
    string columns = "L_ORDERKEY";
    const char *cstr = flatFileName.c_str();

    // Construct file reader
    FileReader* fileReaderLineItem = new FileReader(cstr,columnDelimeter);

    //Read the file into memory
    vector <vector <string> > lineITemData = fileReaderLineItem->ReadData(readRowCount);

    // Get number of rows in the file
    int fileRowCount = fileReaderLineItem->GetRowCount();

    // Init the target table
    string tableName = "LINEITEM";
    Table* lineItemTable = new Table(tableName,fileRowCount);

    // Configure the table schema for a lineitems table
    lineItemTable->InitForLineItem();
    lineItemTable->LoadDataIntoTable(lineITemData);

    // Get Nth column
    Column* l_orderkeyColumns = lineItemTable->GetColumnByName("L_ORDERKEY");
    Column* l_extendedpriceColumn = lineItemTable->GetColumnByName("L_EXTENDEDPRICE");
    Column* l_discountColumn = lineItemTable->GetColumnByName("L_DISCOUNT");
    Column* l_quantity = lineItemTable->GetColumnByName("L_QUANTITY");

    float* taxDataArray = new float[fileRowCount];
    Column* l_tax = lineItemTable->GetColumnByName("L_TAX");
    l_tax->GetCompressedDatas(taxDataArray);

    int* orderkeyArray = (int*) l_orderkeyColumns->GetDataArray();
    float* extendedPriceDataArray = (float*) l_extendedpriceColumn->GetDataArray();
    float* discountDataArray = (float*) l_discountColumn->GetDataArray();
    float* quantityDataArray = (float*) l_quantity->GetDataArray();

    // ColumnUnitTest;
    //Column::ColumnUnitTest("L_ORDERKEY",fileRowCount,lineITemData);

    // Clear the loaded data
    lineITemData.clear();

    // Arrays check
    for( int i = 0; i < 5; i++)
    {
        cout << "<< L_ORDERKEY " << orderkeyArray[i]<< endl;
        cout << " L_EXTENDEDPRICE "<< extendedPriceDataArray[i] <<endl;
        cout << "L_DISCOUNT "<< discountDataArray[i] <<endl;
        cout << "L_TAX "<< taxDataArray[i] <<endl;
        cout << "L_QUANTITY "<< quantityDataArray[i] <<endl;
    }

    // Sum(l_orderkey+l_orderkey)
    int sumOfApBScalar = SumAddTwoArraysScalar(orderkeyArray,orderkeyArray,datasize);
    int64_t sumofApBSimd = SumAddTwoArraysSIMD(orderkeyArray,orderkeyArray,datasize);

    cout << "sumOfApBScalar " << sumOfApBScalar << endl;
    cout << "sumofApBSimd " << sumofApBSimd << endl;

    // (l_extendedprice*(1-l_discount))
    float sumofQ1Scalar = SumMultTwoArraysScalarfloat(extendedPriceDataArray,discountDataArray,datasize,loopCount);
    cout << "Scalar (l_extendedprice*(1-l_discount)) " << sumofQ1Scalar << endl;

    float sumofQ1Simd = SumMultTwoArraysSIMDfloat(extendedPriceDataArray,discountDataArray,datasize,loopCount);
    cout << "SIMD (l_extendedprice*(1-l_discount)) " << sumofQ1Simd << endl;

    //(l_extendedprice*(1-l_discount)*(1+l_tax))
    float sumofQ1Expression2Sacalar = Q1Expression2Scalar(extendedPriceDataArray,discountDataArray,taxDataArray,datasize,loopCount);
    cout << "Scalar (l_extendedprice*(1-l_discount)*(1+l_tax)) " << sumofQ1Expression2Sacalar << endl;

    float sumofQ1Expression2SIMD = Q1Expression2SIMD(extendedPriceDataArray,discountDataArray,taxDataArray,datasize,loopCount);
    cout << "SIMD (l_extendedprice*(1-l_discount)*(1+l_tax)) " << sumofQ1Expression2SIMD << endl;

    clock_t begin , end;
    double elapsed_secs;
    begin = clock();
    Q1Results scalarQ1 =  RunQ1Scalar (quantityDataArray,extendedPriceDataArray,
                           discountDataArray,taxDataArray,datasize,loopCount);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Q1 Scalar finished in  " <<  elapsed_secs <<" seconds"<< endl;
    cout << "Q1 Scalar KRows/sec  " << readRowCount * loopCount /elapsed_secs/1000 <<" seconds"<< endl;
    cout << "Scalar Q1 avg_disc " << scalarQ1.avg_disc << endl;
    cout << "Scalar Q1 avg_price " << scalarQ1.avg_price << endl;
    cout << "Scalar Q1 sum_base_price " << scalarQ1.sum_base_price << endl;

    begin = clock();
    Q1Results simdQ1 =  RunQ1SIMD (quantityDataArray,extendedPriceDataArray,
                           discountDataArray,taxDataArray,datasize,loopCount);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Q1 SIMD finished in  " <<  elapsed_secs <<" seconds"<< endl;
    cout << "Q1 SIMD KRows/sec  " << readRowCount * loopCount /elapsed_secs/1000 <<" seconds"<< endl;
    cout << "SIMD Q1 avg_disc " << simdQ1.avg_disc << endl;
    cout << "SIMD Q1 avg_price " << simdQ1.avg_price << endl;
    cout << "SIMD Q1 sum_base_price " << simdQ1.sum_base_price << endl;

}
