// Micro benchmark : Gradiant Decent for Multi Variate Linear Regression


#include<fstream>
#include<iostream>
#include<sstream>
#include<string>
#include<chrono>
#include<vector>
#include<omp.h>


using namespace std;

vector<string> split(const string &s, char delim) 
{
  stringstream ss(s);
  string item;
  vector<string> tokens;
  while (getline(ss, item, delim)) {
    tokens.push_back(item);
  }
  return tokens;
}

void print_vector(vector<float> theta)
{	
	for (int i=0; i < theta.size(); i++)
	{
		cout<<theta[i]<<"  ";
	}
	cout<<endl;
}

void print_matrix(vector<float> mat, int rows, int cols)
{	
	for (int i=0; i < rows; i++)
	{
		for(int j = 0; j<cols; j++)
		{
			//cout<< "index " <<j + cols*i<<endl;	
			cout<<mat[j + cols*i]<<"  ";
		}
		cout<<endl;
	}
	
}
vector<float> read_data(std::string filename, int * Row_Count, int * Col_Count)
{
	string line;
  	vector<string> line_v;
	
	vector<float> data;
	int size;

	ifstream myfile (filename,std::ifstream::in);
  	
  	if (myfile.is_open())
  	{
  		cout << "File : " << filename <<" loaded succesfully "<< endl;
  		while ( getline (myfile,line) )
    	{
    		line_v = split(line, ',');
    		size = static_cast<int>(line_v.size());
    		
    		for (unsigned i = 0; i < size; ++i) 
    		{
    			float f = strtof((line_v[i]).c_str(),0);
    			if (f != 0)
    			{
    				//cout<<f<<" ";
        			data.push_back(f);
        		}
      		}
      		//cout<<endl;
    	}
  	}
  	else cout << " Unable to open file " << '\n';

  	*Col_Count = size;
  	*Row_Count = data.size()/ *Col_Count;
  	
  	cout<<" File : "<<filename<<" Row count : "<< *Row_Count<<" Col count : "<< *Col_Count<<endl;
  	
  	return data;
}	

vector <float> transform (const float *m, const int R, const int C ) {
    
    /*  Returns a transpose matrix of input matrix.
     Inputs:
     m: vector, input matrix
     C: int, number of columns in the input matrix
     R: int, number of rows in the input matrix
     Output: vector, transpose matrix mT of input matrix m
     */
    
    vector <float> mT (C*R);
    
    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}


vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns) 
{
    vector <float> output (m1_rows*m2_columns, 0);

     // Transpose m2;
    const vector <float> m2t = transform(&m2[0], m2_columns,m1_columns );
    const int m2t_rows = m2_columns;
    const int m2t_columns = m1_columns;
	//print_matrix(m2t, m2t_rows, m2t_columns);
    
	//int row_m2,row_m1,k;

    //#pragma omp parallel num_threads(4) 
    //{
    	int row_m2,row_m1,k;
    	float sum;

      //#pragma omp parallel for num_threads(2) private(row_m2,k,sum)
      for(  row_m1 = 0; row_m1 < m1_rows; row_m1++ ) 
      {
        #pragma omp parallel for private(k,sum)
        for(  row_m2 = 0; row_m2 < m2t_rows; row_m2++ )
        {
           sum = 0.0;
          //#pragma omp parallel for private(row_m1, row_m2)
          for( k = 0; k < m1_columns; k++ ) 
          {
             sum += m1[ row_m1 * m1_columns  + k ] * m2t[ row_m2 * m2t_columns + k ];
          }
          output[ row_m1 * m2t_rows + row_m2 ] = sum;
           
        }
      }
    //}

    return output;
}

vector <float> dot_serial (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns) 
{
    vector <float> output (m1_rows*m2_columns, 0);

    // Transpose m2;
    const vector <float> m2t = transform(&m2[0], m2_columns,m1_columns );
    const int m2t_rows = m2_columns;
    const int m2t_columns = m1_columns;

    // cout<<"m1_rows : "<<m1_rows<<" m2t_rows : "<<m2t_rows<<" m1_columns : "<<m1_columns<<endl;

    //print_matrix(m2t, m2t_rows, m2t_columns);
    //print_vector(m2t);
    float sum;

      
      for( int row_m1 = 0; row_m1 < m1_rows; row_m1++ ) 
      {
        for( int row_m2 = 0; row_m2 < m2t_rows; row_m2++ )
        {
          for( int k = 0; k < m1_columns; k++ ) 
          {
            output[ row_m1 * m2t_rows + row_m2 ]  += m1[ row_m1 * m1_columns  + k ] * m2t[ row_m2 * m2t_columns + k ];
            //cout<<"m1[ row_m1 * m1_columns  + k ]"<<m1[ row_m1 * m1_columns  + k ]<<"m2t[ row_m2 * m2t_columns + k ]"<<m2t[ row_m2 * m2t_columns + k ]<<endl;
          	//cin>>sum;
          }
        }
      }
    

    return output;
}
float Compute_Cost(vector<float> X, int X_Col,int X_Row, vector<float> Y, int Y_Col, int Y_Row , vector<float> theta)
{
	float Cost,Error;
	Cost=0.0;
	vector<float> H,Theta;

	H = dot(theta,X,1,theta.size(),X_Row);
	
	//cout<<H.size()<<endl;

	for(int i=0;i<H.size();i++)
	{
		Error = H[i] - Y[i];

		Cost += (1.0/H.size()) * Error*Error;

	}

	return Cost;
}

vector<float> step_gradient(vector<float> X, int X_Row, int X_Col, vector<float> Y, int Y_Row , int Y_Col, float rate, vector<float> theta)
{

	vector<float> H,Error,Theta,Theta_T;
	
	Error.assign(Y.size(),0.0);
	Theta.assign(theta.size(),0.0);

	vector<float> Gradient,X_T;
	//printf("Col_Len %d \n ",Col_Len);
	//printf("Row_Len %d \n ",Row_Len);
	
	
	//Compute theat_T by just sawping it rows and colum as it is an aarray
	Theta_T = theta;
	const int theta_rows = 1;
    const int theta_columns = X_Row ;


	// Theat_T * X
	H = dot(Theta_T,X,theta_rows,theta_columns,X_Col);

	
	//vector<float> E = H - Y;

	for(int i=0;i<H.size();i++)
	{
		Error[i] = Y[i] - H[i] ;
	}


	X_T = transform (&X[0],X_Col, X_Row);
	int X_T_Rows = X_Col;
	int X_T_Cols = X_Row;

	//print_matrix(X_T,X_T_Rows,X_T_Cols);
	
	//cout<<"X_T_Rows"<<X_T_Rows<<endl;
	//cout<<"X_T_Cols"<<X_T_Cols<<endl;

	Gradient = dot(Error,X_T,1,Error.size(),X_T_Cols);	

	//cout<<"H size"<<H.size()<<endl;
	//print_vector(H);
	//cout<<"Error : "<<endl;
	//print_vector(Error);	
	//cout<<" gradient : "<<endl;
	//print_vector(Gradient);
		
	float c = (-2.0/X_Col)*rate;
	cout<<" c : "<<c<<endl;

	for(int i=0;i<Gradient.size();i++)
	{	
		//cout<<" Theta Before: "<<theta[i]<<endl;	 //<< "2/X_Row)" <<(2.0/X_Row) <<endl;
		//float t =  //(float)(2/X_Row)*
		Theta[i] = theta[i] - c*Gradient[i];;
		//cout<<"G : "<<Gradient[i]<<" Theta after : "<<Theta[i]<<"t : "<<t<<endl;
	}

	return Theta;
}


vector<float> gradient_decent(vector<float> X, int X_Row,int X_Col, vector<float> Y,  int Y_Row, int Y_Col, float rate,int iterations)
{

	//print_matrix(X, X_Row,X_Col);
	//print_vector(X);

	//print_matrix(Y, Y_Row,Y_Col);
	
	vector<float> theta;

	theta.assign(X_Row,0);

	float cost;

	int No_of_Rows = Y.size();
	int No_Of_Col = X.size()/No_of_Rows;

	printf(" No of Features : %d \n ",No_Of_Col);
	printf(" No of Training Samples : %d \n ",No_of_Rows);
	
	cout<<" Starting gradient decent "<<endl;

	for(int i=0;i<iterations;i++)
	{
		theta = step_gradient(X,X_Row,X_Col,Y,Y_Row,Y_Col,rate,theta);
		//cout<<"theta : "<<theta[0]<<theta[1]<<theta[2]<<endl;
		cout<<" theta : "<<endl;
		print_vector(theta);
		cost = Compute_Cost(X,X_Row,X_Col,Y,Y_Row,Y_Col,theta);

		cout <<"Iteartions : "<<i<<" Cost : "<<cost<<endl;

	}

	return theta;
}



int main(int argc, char * argv[])
{
	if (argc < 3) cout <<"Usage : <X_File> <Y_File> <Iterations>"<<endl; 

	//for (int i = 1; i < argc; ++i)
     //   cout << argv[i] << "\n";

	string fileX,fileY;
	int iterations;

	int Row_X,Col_X,Row_Y,Col_Y;

	double start_time, time;

	vector<float> Xdata,Ydata,theta;

	fileX = argv[1];
	cout<<"X Data File: "<<fileX<<endl;

	fileY = argv[2];
	cout<<"Y Data File: "<<fileY<<endl;
	
	iterations = atoi(argv[3]);
	cout<<"Iteartions:  "<<iterations<<endl;
	
	cout<< "Loading data Files" << endl;

	// Xadat is transformed 
	Xdata =  read_data(fileX,&Row_X,&Col_X);


	Ydata =  read_data(fileY,&Row_Y,&Col_Y);
	// Ydata in always 1*n or n*1 
	// to make sure very thng works fine we have to make it 1*n that can be doe by jsut fixing rows to 1
	Row_Y=1;
	Col_Y=Ydata.size();

	//Ydata_t =  transform (Ydata,Col_Y, Row_Y);

	cout << " Starting gradient Decent "<<endl;

	start_time = omp_get_wtime();

	theta = gradient_decent(Xdata,Row_X,Col_X, Ydata,Row_Y,Col_Y,0.01,iterations);

	time = omp_get_wtime() - start_time;

	cout << "OPM Time: " << time << "s" << endl;


}