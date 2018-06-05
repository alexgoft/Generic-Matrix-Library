// ".hpp" file of the template matrix.
#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <thread>

#include "Complex.h"
#include "MatrixException.h"

// Default size of matrix is 1x1.
const int DEFAULT_SIZE = 1;

// The indentation between the values of a printed matrix.
const char INDENTATION = '\t';

// Error messages sent to the exceptions.
const char* ADD_SIZE_ERR = "cannot addition matrices of different sizes.";
const char* SUB_SIZE_ERR = "cannot subtract matrices of different sizes.";
const char* MUL_SIZE_ERR = "cannot calculate multiplication because dims of matrices dont match";
const char* OBJ_ERR = "cannot create new object";
const char* MAT_DIM_ERR = "if 1 of the dimension is 0 than there other had to be 0 too";
const char* VEC_DIM_ERR = "given vector don't match dimensions";
const char* IDX_ERR = "index out of boundaries of matrix";

//-----------------------------------------------------------------------------------------------//

/**
 * This class represents a template matrix.
 */
template <class T> class Matrix
{
public:

///////////////////////////////////CUNSTRUCTOR///////////////////////////////////

/**
 * Default constructor
 */
Matrix<T>();
/** 
 * Matrix size constructor.
 */
Matrix<T>(unsigned int rows, unsigned int cols);
/** 
 * Copy constructor.
 */
Matrix<T>(const Matrix& other);
/** 
 * Move constructor.
 */
Matrix<T>(const Matrix && other);
/** 
 * Construct a matrix via vector.
*/
Matrix<T>(unsigned int rows, unsigned int cols, const std::vector<T>& cells);
/** 
 * Destructor of a matrix.
*/
~Matrix() {}

////////////////////////////////////OPERATORS////////////////////////////////////

//---------------------------------------------------------------------//

/** 
 * Over load of () operator.
 * Grants access to the value (row index, col index)
 */
T& operator ()(unsigned int row, unsigned int col);
/** 
 * Over load of () operator.
 * Grants access to the value (row index, col index) but for const Matrix<T>.
 */
const T& operator()(unsigned int row, unsigned int col) const;

//---------------------------------------------------------------------//

/**
 * The overloaded operator '<<' allow writing needed data to a received stream.
 */
template <class P> 
friend std::ostream& operator<<(std::ostream& os, const Matrix<P>& other);

//---------------------------------------------------------------------//

/** 
 * The matrix on the left side of the equation sign is initialized with size and values of the 
 * matrix on the right size of the equation sign. Values of fields are copied.
 */
Matrix<T>& operator=(const Matrix<T>& other);
/** 
 * Move assignment operator.
 */
Matrix<T>& operator=(const Matrix<T> && other);

//---------------------------------------------------------------------//

/**
 * Subtraction operator - create new matrix and initialize its values by mathematical
 * manners.
 * Throw exception in case the dimensions are incorrect.
 */
const Matrix<T> operator-(const Matrix<T>& other) const;

//--------------------------------ADDITION-----------------------------//

/**
 * This helping function, calculates the addition of two matrices given  a certain row.
 */
static void _additionByRow(unsigned int row, Matrix<T>& result, const Matrix<T>& thisMatrix,\
                           const Matrix<T>& other);
/**
 * Addition operator - create new matrix and initialize its values by mathematical
 * manners.
 * Throw exception in case the dimensions are incorrect.
 */
const Matrix<T> operator+(const Matrix<T>& other) const;

//-------------------------------MULTIPLICATION------------------------//

/**
 * This helping function, calculates the multiplication of two matrices given  a certain row.
 */
static void _multiplicationByRow(unsigned int row, Matrix<T>& result, const Matrix<T>& thisMatrix,\
                                 const Matrix<T>& other);
/**
 * Multiplication operator - create new matrix and initialize its values by mathematical
 * manners.
 * Throw exception in case the dimensions are incorrect.
 */
const Matrix<T> operator*(const Matrix<T>& other) const;

//-------------------------------MULTIPLICATION------------------------//

static void calculateHelper(void (*operation)(unsigned int, Matrix<T>&, const Matrix<T>&, \
                            const Matrix<T>&), Matrix<T>& result,  const Matrix<T>& mat1, \
                            const Matrix<T>& mat2);

//-------------------------------BOOLEAN OPERATORS---------------------//

/**
 * Compare operator. check if two matrices are the same.
 * First by checking if it is the same object. Then, after comparing dimensions, compare values.
 */
bool operator==(const Matrix<T>& other) const;
/**
 * Compare operator. check if two matrices are the same.
 * First by checking if it is the same object. Then, after comparing dimensions, compare values.
 */
bool operator!=(const Matrix<T>& other) const;

//---------------------------------------------------------------------//

////////////////////////////////MATRIX OPERATIONS////////////////////////////////

/**
 * Transpose function: flips the matrix with its values. Implemented by initializing new matrix
 * with opposite dimensions and copied values. 
 */
Matrix<T> trans() const;
/**
 * Simple method that checks if this matrix is square.
 */
bool isSquareMatrix() const; 
/**
 * Trace function - sums all values on the diagonal.
 */
T trace() const;

////////////////////////////////////ITERATORS////////////////////////////////////

/**
 * Using the iterator of the vector.
 */
typedef typename std::vector<T>::const_iterator const_iterator;
/**
 * Returns an iterator pointing to the first element in the vector.
 */
const_iterator begin() { return _matrix.cbegin(); }
/**
 * Returns an iterator referring to the past-the-end element in the vector container.
 */
const_iterator end() { return _matrix.cend(); }

/////////////////////////////////////GETTERS/////////////////////////////////////

/**
 * Getter for the matrix wrapped by Matrix<T>.
 */
std::vector<T>& getMatrix() { return _matrix; }
/** 
 * In case the user work with constant matrices, a const version of the matrix will be 
 * returned.
 */
const std::vector<T>& getMatrix() const { return _matrix; }
/** 
 *Getter for number of rows.
 */
unsigned int rows() const { return _rows; }
/** 
 *Getter for number of rows.
 */
unsigned int cols() const { return _cols; }

/////////////////////////////////////PARALLEL//////////////////////////////////////

// states whether parallel computation is will be used or not.
static bool parallel;

/*
 * Sets the computation way of + and * operator. ( The variable parallel) 
 */
static void setParallel(bool parallel);

////////////////////////////////////VARIABLES////////////////////////////////////

private:
    // Number of rows.
    unsigned int _rows;
    // Number of columns.
    unsigned int _cols;
    // The matrix represented by one dimensional vector.
    std::vector<T> _matrix;
};

/**
 * Default constructor
 */
template<class T>
inline Matrix<T>::Matrix() try: _matrix(DEFAULT_SIZE, T())
{}
catch(std::bad_alloc& ex)
{
    throw MatrixException(OBJ_ERR);
}
/** 
 * Matrix size constructor.
 */
template<class T>
inline Matrix<T>::Matrix(unsigned int rows, unsigned int cols) try: _rows(rows), 
                                                                    _cols(cols), 
                                                                    _matrix(rows*cols, T())
{
    if(((_rows == 0) && (_cols != 0)) || ((_rows != 0) && (_cols == 0)))
    {
        throw MatrixException(MAT_DIM_ERR);
    }
}
catch(std::bad_alloc& ex)
{
    throw MatrixException(OBJ_ERR);
}
/** 
 * Copy constructor.
 */
template<class T>
inline Matrix<T>::Matrix(const Matrix& other) try: _rows(other.rows()),
                                                   _cols(other.cols()), 
                                                   _matrix(other._matrix)
{
    if(((other.rows() == 0) && (other.cols() != 0)) || \
       ((other.rows() != 0) && (other.cols() == 0)))
    {
        throw MatrixException(MAT_DIM_ERR);
    }
}
catch(std::bad_alloc& ex)
{
    throw MatrixException(OBJ_ERR);
}
/** 
 * Move constructor.
 */
template<class T>
inline Matrix<T>::Matrix(const Matrix && other) try: _rows(other.rows()),
                                             _cols(other.cols()),
                                             _matrix(std::move(other._matrix))
{
    if(((other.rows() == 0) && (other.cols() != 0)) || \
       ((other.rows() != 0) && (other.cols() == 0)))
    {
        throw MatrixException(MAT_DIM_ERR);
    }
}
catch(std::bad_alloc& ex)
{
    throw MatrixException(OBJ_ERR);
}
/** 
 * Construct a matrix via vector.
*/
template <class T> 
inline Matrix<T>::Matrix(unsigned int rows, unsigned int cols, \
                         const std::vector<T>& cells) try: _rows(rows), 
                                                           _cols(cols), 
                                                           _matrix(cells)
{
    if(((_rows == 0) && (_cols != 0)) || ((_rows !=0) && (_cols == 0)))
    {
        throw MatrixException(MAT_DIM_ERR);
    }
    // Check if col and row number match the dimension of the vector.
    if((rows*cols) != cells.size())
    {
        throw MatrixException(VEC_DIM_ERR);
    }
}
catch(std::bad_alloc& ex)
{
    throw MatrixException(OBJ_ERR);
}

///////////////////////////////////////////////////////////////////////////////////

// default initialization.
template <typename T>
bool Matrix<T>::parallel = false;

/*
 * Sets the computation way of + and * operator. ( The variable parallel) 
 */
template <typename T>
/**
 * This function sets the variable isParallel.
 */
inline void Matrix<T>::setParallel(bool isParallel)
{
    switch (isParallel) 
    {
        case true:  
            if(parallel == false)
            {
                std::cout << "Generic Matrix mode changed to parallel mode." << std::endl;
                parallel = true;
            }
            break;
        case false: 
            if(parallel == true)
            {
                std::cout << "Generic Matrix mode changed to non-parallel mode." << std::endl;
                parallel = false;
            }
            break;
    }
}

//-----------------------------------------------------------------------------------------------//

/** 
 * Over load of () operator.
 * Grants access to the value (row index, col index)
 */
template <class T> 
inline T& Matrix<T>::operator()(unsigned int row, unsigned int col)
{   
    if((_rows < row) || (_cols < col))
    {
        throw MatrixException(IDX_ERR);
    }
    return _matrix[(row)*_cols + (col)];
}
/** 
 * Over load of () operator.
 * Grants access to the value (row index, col index) but for constant Matrix<T>.
 */
template <class T> 
inline const T& Matrix<T>::operator()(unsigned int row, unsigned int col) const
{
    if((_rows < row) || (_cols < col))
    {
        throw MatrixException(IDX_ERR);
    }
    return _matrix[(row)*_cols + (col)];
}

//-----------------------------------------------------------------------------------------------//

/**
 * The overloaded operator '<<' allow writing needed data to a received stream.
 */
template <class P> 
inline std::ostream& operator<<(std::ostream& os, const Matrix<P>& other)
{  
    unsigned int i, j;

    for (i = 0; i < other.rows(); ++i)
    {   
        for (j = 0; j < other.cols(); ++j)
        {
            // The last integer should not be printed with ' '.
            if(j == other.cols())
            {
                os << other(i, j);
                break;
            }
           os << other(i, j) << '\t';
        }
        os << std::endl;
    }
    
    return os;
}

//-----------------------------------------------------------------------------------------------//

/** 
 * The matrix on the left side of the equation sign is initialized with size and values of the 
 * matrix on the right size of the equation sign. Values of fields are copied.
 */
template <class T> 
inline Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
    if(&other != this)
    {
        _matrix = other.getMatrix();
        _rows = other.rows();
        _cols = other.cols();
    }
    return *this;
}
/** 
 * Move assignment operator.
 */
template <class T> 
inline Matrix<T>& Matrix<T>::operator=(const Matrix<T> && other)
{
    if(&other != this)
    {
        _matrix = std::move(other._matrix);
        _rows = other.rows();
        _cols = other.cols();
    } 

    return *this;
}

//-----------------------------------------------------------------------------------------------//

/**
 * This helping function, calculates the addition of two matrices given  a certain row.
 */
template <class T> 
inline void Matrix<T>::_additionByRow(unsigned int row, Matrix<T>& result, \
                                      const Matrix<T>& thisMatrix, const Matrix<T>& other)
{   
    unsigned int j;

    for (j = 0; j < other.cols(); ++j)
    {
        result(row, j) = thisMatrix(row, j) + other(row, j);
    }
}

/**
 * Addition operator - create new matrix and initialize its values by mathematical
 * manners.
 * Throw exception in case the dimensions are incorrect.
 */
template <class T> 
inline const Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const
{
    if((_rows != other.rows()) || (_cols != other.cols())) 
    {
        throw MatrixException(ADD_SIZE_ERR);
    }
    
    Matrix<T> result(other.rows(), other.cols());

    if(parallel == true)
    {
        calculateHelper(_additionByRow, result, *this, other);
    }
    else
    {   
        unsigned int i;

        for (i = 0; i < other.rows(); ++i)
        {   
            _additionByRow(i, result, *this, other);
        } 
    }
    
    return result;
}

//-----------------------------------------------------------------------------------------------//

/**
 * This helping function, calculates the multiplication of two matrices given  a certain row.
 */
template <class T> 
inline void Matrix<T>::_multiplicationByRow(unsigned int row, Matrix<T>& result, \
                                            const Matrix<T>& thisMatrix, const Matrix<T>& other)
{ 
    unsigned int j, k;
    for (j = 0; j < other._cols; ++j)
    {
        T sum = 0;

        for(k = 0; k < other._rows; ++k)
        {
            sum += thisMatrix(row, k) * other(k, j);
        }

        result(row, j) = sum;
    }
}

/**
 * Multiplication operator - create new matrix and initialize its values by mathematical
 * manners.
 * Throw exception in case the dimensions are incorrect.
 */
template <class T> 
inline const Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const
{
    if(_cols != other.rows())
    {
        throw MatrixException(MUL_SIZE_ERR);
    }

    Matrix<T> result(_rows, other._cols);

    if(parallel == true)
    {
        calculateHelper(_multiplicationByRow, result, *this, other);
    }
    else
    {   
        unsigned int i;

        for (i = 0; i < other.rows(); ++i)
        {   
            _multiplicationByRow(i, result, *this, other);
        } 
    }
    
    return result;
}

//----------------------------------------PARALLEL HELPER----------------------------------------//

template <class T> 
inline void Matrix<T>::calculateHelper(void (*operation)(unsigned int, Matrix<T>&, \
                                       const Matrix<T>&, const Matrix<T>&), Matrix<T>& result, \
                                       const Matrix<T>& mat1, const Matrix<T>& mat2)
{
    std::vector<std::thread> threads;

    unsigned int i;

    // spawn new thread with a given row.
    for (i = 0; i < mat2.rows(); ++i)
    {   
        threads.push_back(std::thread(operation, i, std::ref(result), \
                          std::ref(mat1), std::ref(mat2)));
    }
    // Pause the execution of the method until all threads are finished.
    for (i = 0; i < mat2.rows(); ++i)
    {   
        threads[i].join();
    }
}

//-----------------------------------------------------------------------------------------------//

/**
 * Subtraction operator - create new matrix and initialize its values by mathematical
 * manners.
 * Throw exception in case the dimensions are incorrect.
 */
template <class T> 
inline const Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const
{   
    if((_rows != other.rows()) || (_cols != other.cols())) 
    {
        throw MatrixException(SUB_SIZE_ERR);
    }
    Matrix<T> result(other.rows(), other.cols());

    unsigned int i, j;

    for (i = 0; i < other.rows(); ++i)
    {   
        
        for (j = 0; j < other.cols(); ++j)
        {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    
    return result;
}

//-----------------------------------------------------------------------------------------------//

/**
 * Compare operator. check if two matrices are the same.
 * First by checking if it is the same object. Then, after comparing dimensions, compare values.
 */
template <class T> 
inline bool Matrix<T>::operator==(const Matrix<T>& other) const
{
    // Check if it's the same object.
    if(&other != this) 
    {                
        // Assure dimensions.
        if((_rows != other.rows()) || (_cols != other.cols())) 
        {
            return false;
        }
        // check by value.
        unsigned int i;
        for(i = 0; i < _matrix.size() ; i++)
        {
            if(_matrix[i] != other.getMatrix()[i])
            {
                return false;
            }
        }
    }
    return true;
}
/**
 * Compare operator. check if two matrices are the same.
 * First by checking if it is the same object. Then, after comparing dimensions, compare values.
 */
template <class T> 
inline bool Matrix<T>::operator!=(const Matrix<T>& other) const
{
    return !(*this == other);
}

//-----------------------------------------------------------------------------------------------//

/**
 * Transpose function: flips the matrix with its values. Implemented by initializing new matrix
 * with opposite dimensions and copied values. 
 */
template <class T> 
inline Matrix<T> Matrix<T>::trans() const
{
    Matrix<T> matrix(_cols, _rows);

    unsigned int i, j;

    for (i = 0; i < _rows ; ++i)
    {
        for (j = 0; j < _cols; ++j)
        {
            matrix(j, i) = (*this)(i, j);
        }
    }

    return matrix;
}

/**
 * Specialized version of the method -transpose(): conjugate transpose.
 */
template <> 
inline Matrix<Complex> Matrix<Complex>::trans() const
{
    Matrix<Complex> matrix(_cols, _rows);

    unsigned int i, j;

    for (i = 0; i < _rows ; ++i)
    {
        for (j = 0; j < _cols; ++j)
        {
            matrix(j, i) = (*this)(i, j).conj();
        }
    }

    return matrix;
}
//-----------------------------------------------------------------------------------------------//

/**
 * Simple method that checks if this matrix is square, by comparing the number of cols and rows.
 */
template <class T> 
inline bool Matrix<T>::isSquareMatrix() const
{
    if(_cols == _rows)
    {
        return true;
    }
    return false;
}
/**
 * Trace function - sums all values on the diagonal.
 */
template <class T> 
inline T Matrix<T>::trace() const
{
    T trace = T();
    unsigned int i;

    for (i = 0; i < _rows ; ++i)
    {
        trace += (*this)(i, i);
    }

    return trace;
}

//-----------------------------------------------------------------------------------------------//

#endif // _MATRIX_HPP