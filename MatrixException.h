/**
 * MatrixException.h has the structure of the exception class.
 */
#ifndef _MATRIXEXCEPTION_H
#define _MATRIXEXCEPTION_H

#include <exception>

/**
 * Class that represent an exception that can be caused during run time
 * while working with matrices.
 */
class MatrixException : public std::exception 
{
private:
	// The message.
    const char* _err_msg;

public:

	/**
	 * The constructor of the custom exception class. recieve and store
	 * error message.
	 */
	MatrixException(const char *msg) : _err_msg(msg) {};

	/**
	 * Destructor of the class MatrixException.
	 */
    ~MatrixException() throw() {};

    /**
     * virtual class that's derived from the base class exception.
     * returns the error message for further operations.
     */
    virtual const char *what() const throw() 
    { 
    	return _err_msg;
    };
};

#endif // _MATRIXEXCEPTION_H