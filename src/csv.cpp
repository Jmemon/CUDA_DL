#include "../include/csv.h"
#include <exception>
#include <istream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

/* ----------------------------------------------
removeQuotes

Paramters:
	str - string to remove quotes from
		- passed by ref, so nothing needs to be returned

Assumes strings to be removes are on ends of string
Takes string with quotes within string and removes them
--------------------------------------------- */
void removeQuotesEnd(std::string& str)
{
	if (str.find("\"") == std::string::npos)
		return;

	str.erase(str.begin());
	str.erase(str.end() - 1);
} // end removeQuotes

/* ----------------------------------------------
splitLineIntoTokens

Parameters:
	line - string object; should represent only one line from file
	cols - tells how many elements should be in result vector

Takes line from file and uses the commas as delimiters

Returns:
	result - vector of strings, each string is from one of the columns of csv file
---------------------------------------------- */
std::vector<std::string> splitLineIntoTokens(std::string& line, const int cols)
{
	// --- Error Check -----------------------------
	// npos - not part of string - is returned if substring passed to find() is not found 
	if (line.find('\n') != std::string::npos)
		throw std::invalid_argument("splitLineIntoTokens: String input should not contain any newline characters");
	// ---------------------------------------------

	std::vector<std::string> result;

	std::stringstream		 lineStream(line);
	std::string				 cell;

	// go through each entry of line with comma as delimiter
	while (std::getline(lineStream, cell, ','))
		result.push_back(cell);

	// in case there's a trailing comma
	if (!lineStream && cell.empty() && result.size() < cols)
		result.push_back("");

	// makes sure if result has too few entries, empty values are put in 
	for (int i = result.size(); i < cols; i++)
		result.push_back(""); 

	// if result has too many entries pop_back
	for (int i = result.size(); i > cols; i--)
		result.pop_back();

	return result;
} // end splitLineIntoTokens
