 #include <stdio.h>
 #include <iostream>
 #include <math.h>
 #include "extension.h"
 

 int main(){

    float a = 1.;
    int exponent;

    double mantissa = frexp(a, &exponent);

    std::cout << exponent << std::endl;
    std::cout << mantissa << std::endl;

    std::cout << (mantissa)*pow(2,(exponent)) << std::endl;
    std::cout << "test" << std::endl;
    return 0;
 }