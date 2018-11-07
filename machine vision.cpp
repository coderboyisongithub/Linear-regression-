// machine vision.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<opencv2\opencv.hpp>
#include<math.h>
#include<conio.h>

using namespace cv;
using namespace ml;


float a[]={-1,-3,-3,-4,-5,-6};
class LinearRegression
{
	float intercept,slope,x,p,xmean,ymean,_err__;
	int pcol,rcol;
	bool trained,trainingdata;
public:
	Mat traningdata;
	Mat stdDeviation;
	LinearRegression()
	{
		trained=false;
		trainingdata=false;
		intercept=0;
		slope=0;
		x=0;
		p=0;
		ymean=0;
		xmean=0;
		pcol=1;
		rcol=0;
		_err__=0.0;
	}
	LinearRegression(Mat sample)
	{
		intercept=0;
		slope=0;
		x=0;
		p=0;
		ymean=0;
		xmean=0;
		pcol=1;
		rcol=0;
		_err__=0.0;
		traningdata=Mat(sample.rows,sample.cols,CV_32FC1);
		puts("preparing traning data..");
		for(int r=0;r<sample.rows;r++)
		{
			for(int c=0;c<sample.cols;c++)
			{
				traningdata.at<float>(r,c)=sample.at<float>(r,c);
			}
		}
		puts("done..");
	trained=false;	
	trainingdata=true;


	}
	void feed(Mat T)
	{
		traningdata=Mat(T.rows,T.cols,CV_32FC1);
		
		for(int r=0;r<T.rows;r++)
		{
			for(int c=0;c<T.cols;c++)
			{
				traningdata.at<float>(r,c)=T.at<float>(r,c);
			}

		}
		trainingdata=true;
	}
	void pridict(float input,float &result)
	{
		//check if model is trained or not..
		if(trained==true)
		result=(float)(slope*input+intercept+_err__);	
		else {puts("error mode not trained to pridct..");
		return;}
	}
	void train()
	{
		//lets check do model have training data or not..
		if(trainingdata==false)
		{puts("no training data...");
		return;}
		else
		{
		//calculation of ymean..
		float xnet=0,ynet=0;
		printf("\ntraning data feeded to model..\n initiating training algorithm with training datasets:total rows:%d total col:%d\n",traningdata.rows,traningdata.cols);
		for(int r=0;r<traningdata.rows;r++)
		{
			for(int c=0;c<traningdata.cols;c++)
			{
				if(c==pcol)
				{
					ynet+=traningdata.at<float>(r,c);
				}
				else
					continue;

			}
			

		}
		for(int r=0;r<traningdata.rows;r++)
		{
			for(int c=0;c<traningdata.cols;c++)
			{
				if(c==rcol)
				{
					xnet+=traningdata.at<float>(r,c);
				}
				else 
					continue;

			}
		}
		//evaluation..
		ymean=(float)(ynet/traningdata.rows); 
		xmean=(float)(xnet/traningdata.rows);
		printf("\nymean:%f xmean:%f\n",ymean,xmean);
		std::cout<<"traning data\n"<<traningdata;
		//time to calculate deviation of each data from mean posiotion
		stdDeviation=Mat(traningdata.rows,2,CV_32FC1);//matrix for holding standard deviations;
		
		for(int r=0;r<stdDeviation.rows;r++)
		{
			for(int c=0;c<stdDeviation.cols;c++)
			{
				if(c==rcol)
				{
					stdDeviation.at<float>(r,rcol)=(float)(traningdata.at<float>(r,rcol)-xmean);//(x--xmean)
				}
				else if(c==pcol)
				{
					stdDeviation.at<float>(r,pcol)=(float)(traningdata.at<float>(r,pcol)-ymean);//(yi-ymean)
				}

			}

		}

		std::cout<<"\nsdv\n"<<stdDeviation;
		Mat meanProduct(stdDeviation.rows,2,CV_32FC1);//matrix to keep  of std deviation products i.e (xi-xmean) x (yi-ymean) and (xi-xmean)^2
		float *netxp=(float*)malloc(sizeof(float)),*netyp=(float*)malloc(sizeof(float));//sigma summation..
		*netxp=*netyp=0.0;
		for(int r=0;r<stdDeviation.rows;r++)
		{
			for(int c=0;c<stdDeviation.cols;c++)
			{
				if(c==0)
				{
					meanProduct.at<float>(r,c)=(float)(stdDeviation.at<float>(r,0)*stdDeviation.at<float>(r,1));
					*netxp+=meanProduct.at<float>(r,c);
				}
				else if(c==1)
				{
					meanProduct.at<float>(r,c)=(float)(powf(stdDeviation.at<float>(r,0),2));
					*netyp+=meanProduct.at<float>(r,c);
				}
			}
		}
		std::cout<<"\nmp\n"<<meanProduct<<"\n"<<*netxp<<*netyp;;
		
		//now all data is been calculated 
		//no time to set regression parameters.
		slope=(float)(*netyp/(*netxp));
		intercept=(float)(ymean-(slope*xmean));
		//after setting regression parameters not time to test for pridiction errors..
		float *pridiction=(float*)malloc(sizeof(float)),*sigmaError=(float*)malloc(sizeof(float));
		*sigmaError=0.0;
		*pridiction=0.0;
		for(int r=0;r<traningdata.rows;r++)
		{
			for(int c=0;c<traningdata.cols;c++)
			{
				*pridiction=(float)((slope*traningdata.at<float>(r,rcol))+intercept);
				
				//what is error in pridiction then??

				*sigmaError+=(float)(powf((traningdata.at<float>(r,pcol)-*pridiction),2)); //calculating error and doing its summation to find net error;
			
			}
		}
		_err__=(*sigmaError/traningdata.rows);//mean error..
		puts("model trained..");
		trained=true;///trainig flag..
		info();free(sigmaError);free(pridiction);free(netxp);free(netyp);
		
		traningdata.deallocate();
		stdDeviation.deallocate();

		}
	}
	 void info()
	 {
		 if(trained==true)
		 {
		 printf("\n model is trained with ** regression coffecients:\n intercept:%f\n slope:%f \nError:%f",intercept,slope,_err__);
//printf("\nmodel takes upto:%.3f bytes\n",(float)(sizeof(float)*7+sizeof(int)*2+sizeof(bool)*2));
		 }
		 else
			 puts("model is not trained till now..");
	 }

};
void main()
{
	//implementation... of code...
	Mat t(3,2,CV_32FC1),tt(3,2,CV_32FC1);;
	int count=0;
	for(int R=0;R<t.rows;R++)
	{
		for(int C=0;C<t.cols;C++)
		{
			t.at<float>(R,C)=a[count];
			count++;
		}
	}

	float result;
	LinearRegression model(t);
	model.train();
	model.pridict(-5,result);
	printf("\n%f",result);
	
	_getch();
}
	
