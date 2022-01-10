#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

int dwlicz (char a, char b) {
	int ia = a - '0';
	int ib = b - '0';
	int dwucyfrowa = 10*ia+ib;
return dwucyfrowa;
}

int main (int argc,  char* argv[]) {
  if (argv[1] == 0) {
     cout << "\n<Błąd> Nie podano nazwy pliku wejściowego.\n\n";
    return -1;
  }
  ifstream odczyt (argv[1]);
  if (!odczyt.is_open() ) {
    cout << "\n<Błąd> Plik wejściowy nie dał się otworzyć.\n\n";
    return -2;
  }

  string nazwa (argv[1]);
  nazwa.resize(nazwa.size()-4);
  nazwa+="+_.pdb";
  char p;
  ofstream zapis (nazwa);	
  if (!zapis.is_open() ) {
    cout << "\n<Błąd> Plik wyjściowy nie dał się utworzyć.\n\n";
    return -3;
  }

  string linia;
  	
  while (getline(odczyt, linia)) {

    if (linia[24]!=' ') {

    	if (linia[26]!=' ') { 
    		if (linia[24]=='2' && linia[26]=='B') {linia[25]='1';}
			else if (linia[24]=='4') {
				if (linia[26]=='A')	{linia[25]='9';}
				
				else {
					linia[24]='5';
					if (linia[26]=='B') {linia[25]='0';}
					else if (linia[26]=='C') {linia[25]='1';}
					else if (linia[26]=='D') {linia[25]='2';}
					else if (linia[26]=='E') {linia[25]='3';}
					else if (linia[26]=='F') {linia[25]='4';}
					else if (linia[26]=='G') {linia[25]='5';}
					else if (linia[26]=='H') {linia[25]='6';}
					else if (linia[26]=='I') {linia[25]='7';}			
				}			
			}
			linia[26]=' ';
		}
		else {
			int val=dwlicz(linia[24], linia[25]);
//cout << val << "\n";
//cin >> p;
			string nr = "";
			if (val<=20 && val>=18) {
				nr=to_string(val-1);
			}		
			else if (val<=47 && val>=21) {
				nr=to_string(val+1);
			}
			else if (val>=48) {
				nr=to_string(val+10);
			}
			else {
				nr=to_string(val);			
			}

			linia[24]=char(nr[0]);
			linia[25]=char(nr[1]);		
//cout << linia[24]<< linia[25]<<linia[26] << "\n";
//cin >> p;
		}		
	}

  	zapis << linia << "\n";
  }

  odczyt.close();
  zapis.close();

  return 0;
}
