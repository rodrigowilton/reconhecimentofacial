/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * This file is not intended to be easily readable and contains a number of
 * coding conventions designed to improve portability and efficiency. Do not make
 * changes to this file unless you know what you are doing--modify the SWIG
 * interface file instead.
 * ----------------------------------------------------------------------------- */


#ifndef SWIGJAVA
#define SWIGJAVA
#endif



#ifdef __cplusplus
/* SwigValueWrapper is described in swig.swg */
template<typename T> class SwigValueWrapper {
  struct SwigMovePointer {
    T *ptr;
    SwigMovePointer(T *p) : ptr(p) { }
    ~SwigMovePointer() { delete ptr; }
    SwigMovePointer& operator=(SwigMovePointer& rhs) { T* oldptr = ptr; ptr = 0; delete oldptr; ptr = rhs.ptr; rhs.ptr = 0; return *this; }
  } pointer;
  SwigValueWrapper& operator=(const SwigValueWrapper<T>& rhs);
  SwigValueWrapper(const SwigValueWrapper<T>& rhs);
public:
  SwigValueWrapper() : pointer(0) { }
  SwigValueWrapper& operator=(const T& t) { SwigMovePointer tmp(new T(t)); pointer = tmp; return *this; }
  operator T&() const { return *pointer.ptr; }
  T *operator&() { return pointer.ptr; }
};

template <typename T> T SwigValueInit() {
  return T();
}
#endif

/* -----------------------------------------------------------------------------
 *  This section contains generic SWIG labels for method/variable
 *  declarations/attributes, and other compiler dependent labels.
 * ----------------------------------------------------------------------------- */

/* template workaround for compilers that cannot correctly implement the C++ standard */
#ifndef SWIGTEMPLATEDISAMBIGUATOR
# if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x560)
#  define SWIGTEMPLATEDISAMBIGUATOR template
# elif defined(__HP_aCC)
/* Needed even with `aCC -AA' when `aCC -V' reports HP ANSI C++ B3910B A.03.55 */
/* If we find a maximum version that requires this, the test would be __HP_aCC <= 35500 for A.03.55 */
#  define SWIGTEMPLATEDISAMBIGUATOR template
# else
#  define SWIGTEMPLATEDISAMBIGUATOR
# endif
#endif

/* inline attribute */
#ifndef SWIGINLINE
# if defined(__cplusplus) || (defined(__GNUC__) && !defined(__STRICT_ANSI__))
#   define SWIGINLINE inline
# else
#   define SWIGINLINE
# endif
#endif

/* attribute recognised by some compilers to avoid 'unused' warnings */
#ifndef SWIGUNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define SWIGUNUSED __attribute__ ((__unused__))
#   else
#     define SWIGUNUSED
#   endif
# elif defined(__ICC)
#   define SWIGUNUSED __attribute__ ((__unused__))
# else
#   define SWIGUNUSED
# endif
#endif

#ifndef SWIG_MSC_UNSUPPRESS_4505
# if defined(_MSC_VER)
#   pragma warning(disable : 4505) /* unreferenced local function has been removed */
# endif
#endif

#ifndef SWIGUNUSEDPARM
# ifdef __cplusplus
#   define SWIGUNUSEDPARM(p)
# else
#   define SWIGUNUSEDPARM(p) p SWIGUNUSED
# endif
#endif

/* internal SWIG method */
#ifndef SWIGINTERN
# define SWIGINTERN static SWIGUNUSED
#endif

/* internal inline SWIG method */
#ifndef SWIGINTERNINLINE
# define SWIGINTERNINLINE SWIGINTERN SWIGINLINE
#endif

/* exporting methods */
#if defined(__GNUC__)
#  if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#    ifndef GCC_HASCLASSVISIBILITY
#      define GCC_HASCLASSVISIBILITY
#    endif
#  endif
#endif

#ifndef SWIGEXPORT
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   if defined(STATIC_LINKED)
#     define SWIGEXPORT
#   else
#     define SWIGEXPORT __declspec(dllexport)
#   endif
# else
#   if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#     define SWIGEXPORT __attribute__ ((visibility("default")))
#   else
#     define SWIGEXPORT
#   endif
# endif
#endif

/* calling conventions for Windows */
#ifndef SWIGSTDCALL
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   define SWIGSTDCALL __stdcall
# else
#   define SWIGSTDCALL
# endif
#endif

/* Deal with Microsoft's attempt at deprecating C standard runtime functions */
#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE
#endif

/* Deal with Microsoft's attempt at deprecating methods in the standard C++ library */
#if !defined(SWIG_NO_SCL_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_SCL_SECURE_NO_DEPRECATE)
# define _SCL_SECURE_NO_DEPRECATE
#endif

/* Deal with Apple's deprecated 'AssertMacros.h' from Carbon-framework */
#if defined(__APPLE__) && !defined(__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES)
# define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#endif

/* Intel's compiler complains if a variable which was never initialised is
 * cast to void, which is a common idiom which we use to indicate that we
 * are aware a variable isn't used.  So we just silence that warning.
 * See: https://github.com/swig/swig/issues/192 for more discussion.
 */
#ifdef __INTEL_COMPILER
# pragma warning disable 592
#endif


/* Fix for jlong on some versions of gcc on Windows */
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
  typedef long long __int64;
#endif

/* Fix for jlong on 64-bit x86 Solaris */
#if defined(__x86_64)
# ifdef _LP64
#   undef _LP64
# endif
#endif

#include <jni.h>
#include <stdlib.h>
#include <string.h>


/* Support for throwing Java exceptions */
typedef enum {
  SWIG_JavaOutOfMemoryError = 1, 
  SWIG_JavaIOException, 
  SWIG_JavaRuntimeException, 
  SWIG_JavaIndexOutOfBoundsException,
  SWIG_JavaArithmeticException,
  SWIG_JavaIllegalArgumentException,
  SWIG_JavaNullPointerException,
  SWIG_JavaDirectorPureVirtual,
  SWIG_JavaUnknownError
} SWIG_JavaExceptionCodes;

typedef struct {
  SWIG_JavaExceptionCodes code;
  const char *java_exception;
} SWIG_JavaExceptions_t;


static void SWIGUNUSED SWIG_JavaThrowException(JNIEnv *jenv, SWIG_JavaExceptionCodes code, const char *msg) {
  jclass excep;
  static const SWIG_JavaExceptions_t java_exceptions[] = {
    { SWIG_JavaOutOfMemoryError, "java/lang/OutOfMemoryError" },
    { SWIG_JavaIOException, "java/io/IOException" },
    { SWIG_JavaRuntimeException, "java/lang/RuntimeException" },
    { SWIG_JavaIndexOutOfBoundsException, "java/lang/IndexOutOfBoundsException" },
    { SWIG_JavaArithmeticException, "java/lang/ArithmeticException" },
    { SWIG_JavaIllegalArgumentException, "java/lang/IllegalArgumentException" },
    { SWIG_JavaNullPointerException, "java/lang/NullPointerException" },
    { SWIG_JavaDirectorPureVirtual, "java/lang/RuntimeException" },
    { SWIG_JavaUnknownError,  "java/lang/UnknownError" },
    { (SWIG_JavaExceptionCodes)0,  "java/lang/UnknownError" }
  };
  const SWIG_JavaExceptions_t *except_ptr = java_exceptions;

  while (except_ptr->code != code && except_ptr->code)
    except_ptr++;

  jenv->ExceptionClear();
  excep = jenv->FindClass(except_ptr->java_exception);
  if (excep)
    jenv->ThrowNew(excep, msg);
}


/* Contract support */

#define SWIG_contract_assert(nullreturn, expr, msg) if (!(expr)) {SWIG_JavaThrowException(jenv, SWIG_JavaIllegalArgumentException, msg); return nullreturn; } else


    #include <exception>
    #include <stdexcept>
    static JavaVM *cached_jvm = 0;

    JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
        cached_jvm = jvm;
        return JNI_VERSION_1_6;
    }

    static JNIEnv * JNI_GetEnv() {
        JNIEnv *env;
        jint rc = cached_jvm->GetEnv((void **)&env, JNI_VERSION_1_6);
        if (rc == JNI_EDETACHED)
            throw std::runtime_error("current thread not attached");
        if (rc == JNI_EVERSION)
            throw std::runtime_error("jni version not supported");
        return env;
    }

    #include "swig_api.h"
    

#ifdef __cplusplus
extern "C" {
#endif

SWIGEXPORT jshortArray JNICALL Java_net_dlib_globalJNI_create_1java_1array_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jlong jarg1, jlong jarg2) {
  jshortArray jresult = 0 ;
  int16_t arg1 ;
  size_t arg2 ;
  int16_t *argp1 ;
  jshortArray result;
  
  (void)jenv;
  (void)jcls;
  argp1 = *(int16_t **)&jarg1; 
  if (!argp1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Attempt to dereference null int16_t");
    return 0;
  }
  arg1 = *argp1; 
  arg2 = (size_t)jarg2; 
  {
    try {
      result = java::create_java_array(arg1,arg2);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = result; 
  return jresult;
}


SWIGEXPORT jintArray JNICALL Java_net_dlib_globalJNI_create_1java_1array_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jlong jarg1, jlong jarg2) {
  jintArray jresult = 0 ;
  int32_t arg1 ;
  size_t arg2 ;
  int32_t *argp1 ;
  jintArray result;
  
  (void)jenv;
  (void)jcls;
  argp1 = *(int32_t **)&jarg1; 
  if (!argp1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Attempt to dereference null int32_t");
    return 0;
  }
  arg1 = *argp1; 
  arg2 = (size_t)jarg2; 
  {
    try {
      result = java::create_java_array(arg1,arg2);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = result; 
  return jresult;
}


SWIGEXPORT jlongArray JNICALL Java_net_dlib_globalJNI_create_1java_1array_1_1SWIG_12(JNIEnv *jenv, jclass jcls, jlong jarg1, jlong jarg2) {
  jlongArray jresult = 0 ;
  int64_t arg1 ;
  size_t arg2 ;
  int64_t *argp1 ;
  jlongArray result;
  
  (void)jenv;
  (void)jcls;
  argp1 = *(int64_t **)&jarg1; 
  if (!argp1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Attempt to dereference null int64_t");
    return 0;
  }
  arg1 = *argp1; 
  arg2 = (size_t)jarg2; 
  {
    try {
      result = java::create_java_array(arg1,arg2);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = result; 
  return jresult;
}


SWIGEXPORT jbyteArray JNICALL Java_net_dlib_globalJNI_create_1java_1array_1_1SWIG_13(JNIEnv *jenv, jclass jcls, jchar jarg1, jlong jarg2) {
  jbyteArray jresult = 0 ;
  char arg1 ;
  size_t arg2 ;
  jbyteArray result;
  
  (void)jenv;
  (void)jcls;
  arg1 = (char)jarg1; 
  arg2 = (size_t)jarg2; 
  {
    try {
      result = java::create_java_array(arg1,arg2);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = result; 
  return jresult;
}


SWIGEXPORT jfloatArray JNICALL Java_net_dlib_globalJNI_create_1java_1array_1_1SWIG_14(JNIEnv *jenv, jclass jcls, jfloat jarg1, jlong jarg2) {
  jfloatArray jresult = 0 ;
  float arg1 ;
  size_t arg2 ;
  jfloatArray result;
  
  (void)jenv;
  (void)jcls;
  arg1 = (float)jarg1; 
  arg2 = (size_t)jarg2; 
  {
    try {
      result = java::create_java_array(arg1,arg2);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = result; 
  return jresult;
}


SWIGEXPORT jdoubleArray JNICALL Java_net_dlib_globalJNI_create_1java_1array_1_1SWIG_15(JNIEnv *jenv, jclass jcls, jdouble jarg1, jlong jarg2) {
  jdoubleArray jresult = 0 ;
  double arg1 ;
  size_t arg2 ;
  jdoubleArray result;
  
  (void)jenv;
  (void)jcls;
  arg1 = (double)jarg1; 
  arg2 = (size_t)jarg2; 
  {
    try {
      result = java::create_java_array(arg1,arg2);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1crit_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jshortArray jarg1) {
  jint jresult = 0 ;
  java::array_view_crit< int16_t > *arg1 = 0 ;
  java::array_view_crit< int16_t > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum_crit((java::array_view_crit< int16_t > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jshortArray jarg1) {
  jint jresult = 0 ;
  java::array_view< int16_t > *arg1 = 0 ;
  java::array_view< int16_t > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum((java::array_view< int16_t > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1crit_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jshortArray jarg1) {
  java::array_view_crit< int16_t > *arg1 = 0 ;
  java::array_view_crit< int16_t > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign_crit(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1_1SWIG_10(JNIEnv *jenv, jclass jcls, jshortArray jarg1) {
  java::array_view< int16_t > *arg1 = 0 ;
  java::array_view< int16_t > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1crit_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jintArray jarg1) {
  jint jresult = 0 ;
  java::array_view_crit< int32_t > *arg1 = 0 ;
  java::array_view_crit< int32_t > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum_crit((java::array_view_crit< int32_t > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jintArray jarg1) {
  jint jresult = 0 ;
  java::array_view< int32_t > *arg1 = 0 ;
  java::array_view< int32_t > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum((java::array_view< int32_t > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1crit_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jintArray jarg1) {
  java::array_view_crit< int32_t > *arg1 = 0 ;
  java::array_view_crit< int32_t > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign_crit(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1_1SWIG_11(JNIEnv *jenv, jclass jcls, jintArray jarg1) {
  java::array_view< int32_t > *arg1 = 0 ;
  java::array_view< int32_t > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1crit_1_1SWIG_12(JNIEnv *jenv, jclass jcls, jlongArray jarg1) {
  jint jresult = 0 ;
  java::array_view_crit< int64_t > *arg1 = 0 ;
  java::array_view_crit< int64_t > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum_crit((java::array_view_crit< int64_t > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1_1SWIG_12(JNIEnv *jenv, jclass jcls, jlongArray jarg1) {
  jint jresult = 0 ;
  java::array_view< int64_t > *arg1 = 0 ;
  java::array_view< int64_t > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum((java::array_view< int64_t > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1crit_1_1SWIG_12(JNIEnv *jenv, jclass jcls, jlongArray jarg1) {
  java::array_view_crit< int64_t > *arg1 = 0 ;
  java::array_view_crit< int64_t > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign_crit(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1_1SWIG_12(JNIEnv *jenv, jclass jcls, jlongArray jarg1) {
  java::array_view< int64_t > *arg1 = 0 ;
  java::array_view< int64_t > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1crit_1_1SWIG_13(JNIEnv *jenv, jclass jcls, jbyteArray jarg1) {
  jint jresult = 0 ;
  java::array_view_crit< char > *arg1 = 0 ;
  java::array_view_crit< char > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum_crit((java::array_view_crit< char > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT jint JNICALL Java_net_dlib_globalJNI_sum_1_1SWIG_13(JNIEnv *jenv, jclass jcls, jbyteArray jarg1) {
  jint jresult = 0 ;
  java::array_view< char > *arg1 = 0 ;
  java::array_view< char > temp1 ;
  int result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (int)sum((java::array_view< char > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jint)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1crit_1_1SWIG_13(JNIEnv *jenv, jclass jcls, jbyteArray jarg1) {
  java::array_view_crit< char > *arg1 = 0 ;
  java::array_view_crit< char > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign_crit(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1_1SWIG_13(JNIEnv *jenv, jclass jcls, jbyteArray jarg1) {
  java::array_view< char > *arg1 = 0 ;
  java::array_view< char > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT jdouble JNICALL Java_net_dlib_globalJNI_sum_1crit_1_1SWIG_14(JNIEnv *jenv, jclass jcls, jdoubleArray jarg1) {
  jdouble jresult = 0 ;
  java::array_view_crit< double > *arg1 = 0 ;
  java::array_view_crit< double > temp1 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (double)sum_crit((java::array_view_crit< double > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT jdouble JNICALL Java_net_dlib_globalJNI_sum_1_1SWIG_14(JNIEnv *jenv, jclass jcls, jdoubleArray jarg1) {
  jdouble jresult = 0 ;
  java::array_view< double > *arg1 = 0 ;
  java::array_view< double > temp1 ;
  double result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, false); 
  }
  {
    try {
      result = (double)sum((java::array_view< double > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jdouble)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1crit_1_1SWIG_14(JNIEnv *jenv, jclass jcls, jdoubleArray jarg1) {
  java::array_view_crit< double > *arg1 = 0 ;
  java::array_view_crit< double > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign_crit(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1_1SWIG_14(JNIEnv *jenv, jclass jcls, jdoubleArray jarg1) {
  java::array_view< double > *arg1 = 0 ;
  java::array_view< double > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT jfloat JNICALL Java_net_dlib_globalJNI_sum_1crit_1_1SWIG_15(JNIEnv *jenv, jclass jcls, jfloatArray jarg1) {
  jfloat jresult = 0 ;
  java::array< float > arg1 ;
  float result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = java::array<float>(jarg1); 
  }
  {
    try {
      result = (float)sum_crit(arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jfloat)result; 
  return jresult;
}


SWIGEXPORT jfloat JNICALL Java_net_dlib_globalJNI_sum_1_1SWIG_15(JNIEnv *jenv, jclass jcls, jfloatArray jarg1) {
  jfloat jresult = 0 ;
  java::array< float > *arg1 = 0 ;
  java::array< float > temp1 ;
  float result;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    *(arg1) = java::array<float>(jarg1); 
  }
  {
    try {
      result = (float)sum((java::array< float > const &)*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  jresult = (jfloat)result; 
  return jresult;
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1crit_1_1SWIG_15(JNIEnv *jenv, jclass jcls, jfloatArray jarg1) {
  java::array_view_crit< float > *arg1 = 0 ;
  java::array_view_crit< float > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    arg1->reset(jenv, jarg1, true); 
  }
  {
    try {
      assign_crit(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT void JNICALL Java_net_dlib_globalJNI_assign_1_1SWIG_15(JNIEnv *jenv, jclass jcls, jfloatArray jarg1) {
  java::array< float > *arg1 = 0 ;
  java::array< float > temp1 ;
  
  (void)jenv;
  (void)jcls;
  {
    arg1 = &temp1; 
  }
  {
    *(arg1) = java::array<float>(jarg1); 
  }
  {
    try {
      assign(*arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return ;
    }
  }
}


SWIGEXPORT jintArray JNICALL Java_net_dlib_globalJNI_make_1an_1array(JNIEnv *jenv, jclass jcls, jlong jarg1) {
  jintArray jresult = 0 ;
  size_t arg1 ;
  java::array< int32_t > result;
  
  (void)jenv;
  (void)jcls;
  arg1 = (size_t)jarg1; 
  {
    try {
      result = make_an_array(arg1);
    } catch(std::exception& e) {
      jclass clazz = jenv->FindClass("java/lang/Exception");
      jenv->ThrowNew(clazz, e.what());
      return 0;
    }
  }
  {
    jresult = result;
  }
  return jresult;
}


#ifdef __cplusplus
}
#endif

