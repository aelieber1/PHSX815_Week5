// C++ input/output includes
#include <iostream>
#include <fstream>

// ROOT includes (for graphs and plotting)
#include "TGraph.h"
#include "TF1.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMath.h"
#include "TLegend.h"
#include "TAxis.h"
#include "TRandom.h"

// our local includes
#include "Random.hh"

using namespace std;

double bin_width;
double Xmin = -3.;
double Xmax =  3.;
Random random_gen;

// Normal distribution with mean zero and sigma = 1
// Note: multiply by bin width to match histogram
double Gaus(double x){
  return 1./sqrt(2.*acos(-1))*exp(-x*x/2.);
}
double PlotGaus(double* x, double* par){
  return bin_width*Gaus(x[0]);
}

// Uniform (flat) distribution scaled to Gaussian max
// Note: multiply by bin width to match histogram
double Flat(double x){
  return 1./sqrt(2.*acos(-1));
}
double PlotFlat(double* x, double* par){
  return bin_width*Flat(x[0]);
}
// Get a random X value according to flat distribution
double SampleFlat(){
  return Xmin + (Xmax-Xmin)*random_gen.rand();
}

// Piecewise flat with exponential tails, scaled to Gaussian max
// Note: multiply by bin width to match histogram
double FlatPlusExpo(double x){
  if(x <= Xmin/4.)
    return 1./sqrt(2.*acos(-1))*exp(Xmin*Xmin/8.)*exp(-Xmin/2.*x);
  if(x >= Xmax/4.)
    return 1./sqrt(2.*acos(-1))*exp(Xmax*Xmax/8.)*exp(-Xmax/2.*x);
  return 1./sqrt(2.*acos(-1));
}
double PlotFlatPlusExpo(double* x, double* par){
  return bin_width*FlatPlusExpo(x[0]);
}
// Get a random X value according to flat plus exponentials distribution
double SampleFlatPlusExpo(){
  double R = random_gen.rand();
  double Nexp = sqrt(2./acos(-1))/Xmax*(1 - exp(-3.*Xmax*Xmax/8.));
  double Ntot = 2.*Nexp + Xmax/2./sqrt(2.*acos(-1));
  if(R <= Nexp/Ntot){
    double F = random_gen.rand();
    return 2./Xmax*log(F*exp(-Xmax*Xmax/2.)+(1.-F)*exp(-Xmax*Xmax/8));
  }
  if(R >= (1.-Nexp/Ntot)){
    double F = random_gen.rand();
    return -2./Xmax*log(F*exp(-Xmax*Xmax/2.)+(1.-F)*exp(-Xmax*Xmax/8));
  }
  
  return Xmin/4. + (Xmax-Xmin)/4.*random_gen.rand();
}

// main function for compiled executable CookieHypoTest
int main(int argc, char* argv[]){
  bool printhelp = false;

  int Nsample = 0;
  bool dolog = false;
  bool doexpo = false;
  
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
    if(strncmp(argv[i],"-range", 6) == 0){
      i++;
      double range = std::stod(argv[i]);
      if(range > 0){
	Xmax = range;
	Xmin = -range;
      }
    }
    if(strncmp(argv[i],"--log", 5) == 0){
      dolog = true;
    }
    if(strncmp(argv[i],"--expo", 6) == 0){
      doexpo = true;
    }
  }

  // print the executable usage options if the user adds -h or --help 
  if(printhelp){
    cout << "Usage: " << argv[0] << " [options]" << endl;
    cout << "  options:" << endl;
    cout << "   --help(-h)          print options" << endl;
    cout << "   -Nsample [number]   number of random samples" << endl;
    cout << "   -range [number]     x-axis range (# of sigma)" << endl;
    cout << "   --log               do log scale" << endl;
    cout << "   --expo              use exponential tails" << endl;

    return 0;
  }
  
  double Ntrial  = 0;

  TH1D* hist = new TH1D("hist", "hist", 100, Xmin, Xmax);
  bin_width = hist->GetXaxis()->GetBinWidth(1);

  double X, ran, R;
  for(int i = 0; i < Nsample; i++){
    Ntrial += 1;
    if(!doexpo){
      X = SampleFlat();
      R = Gaus(X)/Flat(X);
    } else {
      X = SampleFlatPlusExpo();
      R = Gaus(X)/FlatPlusExpo(X);
    }
    ran = random_gen.rand();
    if(ran > R){ // reject
      i -= 1; // decrease i and try again
      continue;
    } else { // accept
      hist->Fill(X);
    }
  }

  if(hist->Integral() > 0)
    hist->Scale(1./hist->Integral());
  
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


  TCanvas* can0 = (TCanvas*) new TCanvas("can0", "can0", 450, 400);
  double hlo = 0.15;
  double hhi = 0.08;
  double hbo = 0.15;
  double hto = 0.07;
  can0->SetLeftMargin(hlo);
  can0->SetRightMargin(hhi);
  can0->SetBottomMargin(hbo);
  can0->SetTopMargin(hto);
  can0->SetGridx();
  can0->SetGridy();
  if(dolog)
    can0->SetLogy();
  can0->Draw();
  can0->cd();
  
  hist->SetLineColor(kBlue+2);
  hist->SetFillColor(kBlue);
  hist->SetFillStyle(3004);
  hist->Draw("hist");
  hist->GetXaxis()->CenterTitle();
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.05);
  hist->GetXaxis()->SetTitleOffset(1.1);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetXaxis()->SetLabelSize(0.04);
  hist->GetXaxis()->SetTitle("X");
  hist->GetXaxis()->SetTickSize(0.);
  hist->GetYaxis()->CenterTitle();
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleSize(0.05);
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(0.035);
  hist->GetYaxis()->SetTitle("Probability / bin");
  if(!dolog)
    hist->GetYaxis()->SetRangeUser(std::min(bin_width*Gaus(Xmax), 1./double(Nsample+1)),
				   1.5*std::max(hist->GetMaximum(), bin_width*Gaus(0)));
  else
    hist->GetYaxis()->SetRangeUser(std::min(bin_width*Gaus(Xmax), 1./double(Nsample+1)),
				   80*std::max(hist->GetMaximum(), bin_width*Gaus(0)));

  TF1 *gaus = new TF1("mygaus", PlotGaus, Xmin, Xmax);
  gaus->SetLineColor(kGreen+2);
  gaus->SetLineWidth(2);
  gaus->Draw("same");

  TF1 *target;
  if(!doexpo)
    target = (TF1*) new TF1("myflat", PlotFlat, Xmin, Xmax);
  else
    target = (TF1*) new TF1("myflatpexp", PlotFlatPlusExpo, Xmin, Xmax);
  target->SetLineColor(kRed+2);
  target->SetLineWidth(2);
  target->Draw("same");

  TLegend* leg = new TLegend(0.17,0.72,0.3946,0.925);
  leg->SetTextFont(132);
  leg->SetTextSize(0.045);
  leg->SetFillColor(kWhite);
  leg->SetLineColor(kWhite);
  leg->SetShadowColor(kWhite);
  leg->AddEntry(gaus, "target f(x)" , "l");
  leg->AddEntry(target, "proposal g(x)", "l");
  if(hist->Integral() > 0)
    leg->AddEntry(hist, "samples from f(x)");
  leg->Draw();

  can0->SaveAs("RandomGauss.pdf");

  if(Ntrial > 0)
    cout << "Efficiency was " << double(Nsample)/Ntrial << endl;
}

