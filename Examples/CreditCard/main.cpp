#include "../../include/NeuralNet.h"
#include "../../include/csv.h"
#include <iostream>
#include <istream>
#include <fstream>
#include <vector>

// Dataset Info: https://www.kaggle.com/sakshigoyal7/credit-card-customers

std::ostream& operator<< (std::ostream& cout, std::vector<double> &x);

std::vector<std::vector<double> > dataToVect(std::ifstream& fin);

int main(int argc, char *argv[]) {

	std::vector<std::vector<double> > data;
	std::vector<std::vector<double> > x, y;

	// Get Raw Data
	std::ifstream file("BankChurners.csv");
	data = dataToVect(file);
	file.close();

	// make vect of column 2 (age) and column 6 (marital status) 
	// these are values to predict  
	// using these values bc they dont imply each other
	// might have to change network a bit bc 
	// 	classification might make more sense for marital status
	for (int i = 0; i < data.size(); i++)
	{
		std::vector<double> tmp(2);
		tmp[0] = data[i][2];
		tmp[1] = data[i][6];
		y.push_back(tmp);
	} // end for

	// Note we are not using col 0 here; we don't care about CLIENTNUM
	// Same for col 2 and 6; thats what were predicting
	for (int i = 0; i < data.size(); i++)
	{
		std::vector<double> tmp(21);
		for (int j = 0; j < data[i].size(); j++)
		{
			if (j == 0)
				continue;
			else if (j == 1)
				tmp[0] = data[i][j];
			else if (j == 2)
				continue;
			else if (2 < j && j < 6)
				tmp[j - 2] = data[i][j];
			else if (j == 6)
				continue;
			else
				tmp[j - 3] = data[i][j];

		} // end for

	} // end for

	// Output layer has 2 neurons
	// Input layer has 20 neurons
	
	std::vector<int> layers(4);
	layers[0] = 20;
	layers[1] = 300;
	layers[2] = 100;
	layers[3] = 2;

	std::vector<Activation> funcs(3);
	funcs[0] = exponential; 
	funcs[1] = sigmoid;
	funcs[2] = relu;

	Loss err = mse;

	NeuralNet nn(layers, funcs, err);

	return 0;
}

std::ostream& operator<< (std::ostream& cout, std::vector<double> &x)
{
	for (int i = 0; i < x.size(); i++)
		std::cout << x[i] << "  ";	

	std::cout << std::endl;

	return cout;
} // end operator<<

std::vector<std::vector<double> > dataToVect(std::ifstream& fin)
{
	std::vector<std::vector<double> > out;
	std::vector<double> tmp(23);

	std::vector<std::string> interm;
	std::string line;

	std::getline(fin, line); // throw away first line with col labels

	while (std::getline(fin, line))
	{
		interm = splitLineIntoTokens(line, 23);

		for (int i = 0; i < interm.size(); i++)
		{
			removeQuotesEnd(interm[i]);
			
			switch (i)
			{
				case 1:
					if (interm[1] == "Existing Customer")
						tmp[1] = 1;
					else if (interm[1] == "Attrited Customer")
						tmp[1] = 2;
					else
						tmp[1] = 0;
					break;
				case 3:
					if (interm[3] == "M")
						tmp[3] = 1;
					else if (interm[3] == "F")
						tmp[3] = 2;
					else
						tmp[3] = 0;
					break;
				case 5:
					if (interm[5] == "Noneducated")
						tmp[5] = 1;
					else if (interm[5] == "High School")
						tmp[5] = 2;
					else if (interm[5] == "College")
						tmp[5] = 3;
					else if (interm[5] == "Graduate")
						tmp[5] = 4;
					else if (interm[5] == "Post-Graduate")
						tmp[5] = 5;
					else if (interm[5] == "Doctorate")
						tmp[5] = 6;
					else if (interm[5] == "Unknown")
						tmp[5] = 7;
					else
						tmp[5] = 0;
					break;
				case 6:
					if (interm[6] == "Single")
						tmp[6] = 1;
					else if (interm[6] == "Married")
						tmp[6] = 2;
					else if (interm[6] == "Divorced")
						tmp[6] = 3;
					else if (interm[6] == "Unknown")
						tmp[6] = 4;
					else
						tmp[6] = 0;
					break;
				case 7:
					if (interm[7] == "Less than $40K")
						tmp[7] = 1;
					else if (interm[7] == "$40K - $60K")
						tmp[7] = 2;
					else if (interm[7] == "$60K - $80K")
						tmp[7] = 3;
					else if (interm[7] == "$80K - $120K")
						tmp[7] = 4;
					else if (interm[7] == "$120K +")
						tmp[7] = 5;
					else 
						tmp[7] = 0;
					break;
				case 8:
					if (interm[8] == "Blue")
						tmp[8] = 1;
					else if (interm[8] == "Silver")
						tmp[8] = 2;
					else if (interm[8] == "Gold")
						tmp[8] = 3;
					else if (interm[8] == "Platinum")
						tmp[8] = 4;
					else
						tmp[8] = 0;
					break;
				default:
					tmp[i] = std::stoi(interm[i]);

			} // end switch

		} // end for

		out.push_back(tmp);

	} // end while

	return out;
} // end dataToVect
