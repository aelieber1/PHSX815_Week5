// C++ input/output includes
#include <iostream>
#include <fstream>

// ROOT includes (for graphs and plotting)
#include "TGraph.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMath.h"
#include "TLegend.h"
#include "TAxis.h"

// our local includes
#include "Random.hh"

using namespace std;

// main function for compiled executable CookieHypoTest
int main(int argc, char* argv[]){
  bool printhelp = false;

  int Nsample = 1000;
  
  // parsing any command line options added by the user
  for(int i = 1; i < argc; i++){
    if(strncmp(argv[i],"--help", 6) == 0){
      printhelp = true;
      continue;
    }
    if(strncmp(argv[i],"-h", 2) == 0){
      printhelp = true;
      continue;
    }
    if(strncmp(argv[i],"-Nsample", 8) == 0){
      i++;
      int N = std::stoi(argv[i]);
      if(N > 0)
	Nsample = N;
    }
  }

  // print the executable usage options if the user adds -h or --help 
  if(printhelp){
    cout << "Usage: " << argv[0] << " [options]" << endl;
    cout << "  options:" << endl;
    cout << "   --help(-h)          print options" << endl;
    cout << "   -Nsample [number]   number of random samples" << endl;

    return 0;
  }

  double Naccept = 0;
  double Ntotal  = 0;

  // accepted X/Y values
  vector<double> Xaccept;
  vector<double> Yaccept;

  // rejected X/Y values
  vector<double> Xreject;
  vector<double> Yreject;
  
  // sample number
  vector<double> isample;
  // calculated value of Pi (per sample)
  vector<double> calcPi;

  Random random;

  double X, Y;
  int idraw = std::max(1, int(Nsample)/100000);
  for(int i = 0; i < Nsample; i++){
    X = random.rand();
    Y = random.rand();

    Ntotal += 1;
    if(X*X + Y*Y <= 1){ // accept
      Naccept += 1;
      if(i%idraw == 0){
	Xaccept.push_back(X);
	Yaccept.push_back(Y);
      }
    } else { //reject
      if(i%idraw == 0){
	Xreject.push_back(X);
	Yreject.push_back(Y);
      }
    }
    if(i%idraw == 0){
      isample.push_back(Ntotal);
      calcPi.push_back(4.*Naccept/Ntotal);
    }
  }

  // some formating settings
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);

  // for drawing text on canvas
  TLatex l;
  l.SetTextFont(42);
  l.SetTextAlign(21);
  l.SetTextSize(0.04);
  l.SetTextColor(kGreen+2);

  // for drawing line on canvas
  TLine* line = new TLine();
  line->SetLineWidth(4);
  line->SetLineColor(kGreen+2);

  // plot calculated pi vs. sample number
  TGraph* gr_pi = new TGraph(int(isample.size()), &isample[0], &calcPi[0]);

  TCanvas* can0 = (TCanvas*) new TCanvas("can0", "can0", 450, 400);
  double hlo = 0.13;
  double hhi = 0.08;
  double hbo = 0.15;
  double hto = 0.07;
  can0->SetLeftMargin(hlo);
  can0->SetRightMargin(hhi);
  can0->SetBottomMargin(hbo);
  can0->SetTopMargin(hto);
  can0->SetGridx();
  can0->SetGridy();
  can0->Draw();
  can0->cd();

  gr_pi->SetLineColor(kBlue+2);
  gr_pi->SetMarkerColor(kBlue+2);
  gr_pi->Draw("ACP");
  gr_pi->GetXaxis()->CenterTitle();
  gr_pi->GetXaxis()->SetTitleFont(42);
  gr_pi->GetXaxis()->SetTitleSize(0.05);
  gr_pi->GetXaxis()->SetTitleOffset(1.1);
  gr_pi->GetXaxis()->SetLabelFont(42);
  gr_pi->GetXaxis()->SetLabelSize(0.04);
  gr_pi->GetXaxis()->SetTitle("Sample Number");
  gr_pi->GetXaxis()->SetTickSize(0.);
  gr_pi->GetYaxis()->CenterTitle();
  gr_pi->GetYaxis()->SetTitleFont(42);
  gr_pi->GetYaxis()->SetTitleSize(0.05);
  gr_pi->GetYaxis()->SetTitleOffset(1.1);
  gr_pi->GetYaxis()->SetLabelFont(42);
  gr_pi->GetYaxis()->SetLabelSize(0.035);
  gr_pi->GetYaxis()->SetTitle("Approximate #pi");
  gr_pi->GetXaxis()->SetRangeUser(0., isample[isample.size()-1]);

  l.DrawLatex(isample[isample.size()-1]/2., acos(-1)+0.02, "true #pi");
  line->DrawLine(isample[0], acos(-1), isample[isample.size()-1], acos(-1));
  gr_pi->Draw("CP same");

  can0->SaveAs("CalculatedPi.pdf");

  // plot accept/reject points
  TGraph* gr_accept = new TGraph(int(Xaccept.size()), &Xaccept[0], &Yaccept[0]);
  TGraph* gr_reject = new TGraph(int(Xreject.size()), &Xreject[0], &Yreject[0]);
  
  TCanvas* can1 = (TCanvas*) new TCanvas("can1", "can1", 450, 400);
  can1->SetLeftMargin(hlo);
  can1->SetRightMargin(hhi);
  can1->SetBottomMargin(hbo);
  can1->SetTopMargin(hto);
  can1->Draw();
  can1->cd();

  double boxx[4] = {0, 0, 1, 1};
  double boxy[4] = {0, 1, 0, 1};
  TGraph* gr_box = new TGraph(4, boxx, boxy);
  
  gr_accept->SetMarkerColor(kGreen+2);
  gr_accept->SetMarkerStyle(8);
  gr_reject->SetMarkerColor(kRed+2);
  gr_reject->SetMarkerStyle(8);
  gr_box->Draw("AP");
  gr_box->GetXaxis()->CenterTitle();
  gr_box->GetXaxis()->SetTitleFont(42);
  gr_box->GetXaxis()->SetTitleSize(0.05);
  gr_box->GetXaxis()->SetTitleOffset(1.1);
  gr_box->GetXaxis()->SetLabelFont(42);
  gr_box->GetXaxis()->SetLabelSize(0.04);
  gr_box->GetXaxis()->SetTitle("X");
  gr_box->GetXaxis()->SetTickSize(0.);
  gr_box->GetYaxis()->CenterTitle();
  gr_box->GetYaxis()->SetTitleFont(42);
  gr_box->GetYaxis()->SetTitleSize(0.05);
  gr_box->GetYaxis()->SetTitleOffset(1.1);
  gr_box->GetYaxis()->SetLabelFont(42);
  gr_box->GetYaxis()->SetLabelSize(0.035);
  gr_box->GetYaxis()->SetTitle("Y");
  gr_box->GetXaxis()->SetRangeUser(0., 1.);
  gr_box->GetYaxis()->SetRangeUser(0., 1.);

  TF1 *circle = new TF1("circle", "sqrt(1-x*x)", 0, 1);
  circle->SetLineColor(kBlue+2);
  circle->SetLineWidth(2);
  circle->Draw("same");
  gr_accept->Draw("P same");
  gr_reject->Draw("P same");

  can1->SaveAs("CircleQuad.pdf");
}

