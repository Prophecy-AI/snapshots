#include <iostream>
#include <cstdlib>
#include <string>

int main() {
    std::string s = "s1.23";
    double d = std::atof(s.c_str());
    std::cout << "atof('s1.23') = " << d << std::endl;
    return 0;
}
