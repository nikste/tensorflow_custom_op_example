// example.h
#ifndef ZERO_OUT_H_
#define ZERO_OUT_H_

template <typename Device, typename T>
struct ZeroOutFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#endif ZERO_OUT_H_
