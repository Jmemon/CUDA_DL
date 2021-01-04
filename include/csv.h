#ifndef CSV_H
#define CSV_H

#include <istream>
#include <vector>
#include <string>

void removeQuotesEnd(std::string& str);

std::vector<std::string> splitLineIntoTokens(std::string& str, int cols);

#endif // CSV_H
