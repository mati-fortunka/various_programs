#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

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
  nazwa+="+_.txt";

  ofstream zapis (nazwa);	
  if (!zapis.is_open() ) {
    cout << "\n<Błąd> Plik wyjściowy nie dał się utworzyć.\n\n";
    return -3;
  }

  string linia;
  
  //int i=0;	
  while (getline(odczyt, linia)) {
    //i++;
    if (linia[0]=='A') linia.insert(16, " ");
    zapis << linia << "\n";
  }

  odczyt.close ();
  zapis.close ();

  return 0;
}
