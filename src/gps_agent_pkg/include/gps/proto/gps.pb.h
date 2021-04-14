// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: gps.proto

#ifndef PROTOBUF_gps_2eproto__INCLUDED
#define PROTOBUF_gps_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace gps {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_gps_2eproto();
void protobuf_AssignDesc_gps_2eproto();
void protobuf_ShutdownFile_gps_2eproto();

class Sample;

enum SampleType {
  ACTION = 0,
  CUR_LOC = 1,
  PAST_OBJ_VAL_DELTAS = 2,
  PAST_GRADS = 3,
  CUR_GRAD = 4,
  PAST_LOC_DELTAS = 5
};
bool SampleType_IsValid(int value);
const SampleType SampleType_MIN = ACTION;
const SampleType SampleType_MAX = PAST_LOC_DELTAS;
const int SampleType_ARRAYSIZE = SampleType_MAX + 1;

const ::google::protobuf::EnumDescriptor* SampleType_descriptor();
inline const ::std::string& SampleType_Name(SampleType value) {
  return ::google::protobuf::internal::NameOfEnum(
    SampleType_descriptor(), value);
}
inline bool SampleType_Parse(
    const ::std::string& name, SampleType* value) {
  return ::google::protobuf::internal::ParseNamedEnum<SampleType>(
    SampleType_descriptor(), name, value);
}
// ===================================================================

class Sample : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:gps.Sample) */ {
 public:
  Sample();
  virtual ~Sample();

  Sample(const Sample& from);

  inline Sample& operator=(const Sample& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Sample& default_instance();

  void Swap(Sample* other);

  // implements Message ----------------------------------------------

  inline Sample* New() const { return New(NULL); }

  Sample* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Sample& from);
  void MergeFrom(const Sample& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(Sample* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional uint32 T = 1;
  bool has_t() const;
  void clear_t();
  static const int kTFieldNumber = 1;
  ::google::protobuf::uint32 t() const;
  void set_t(::google::protobuf::uint32 value);

  // optional uint32 dX = 2;
  bool has_dx() const;
  void clear_dx();
  static const int kDXFieldNumber = 2;
  ::google::protobuf::uint32 dx() const;
  void set_dx(::google::protobuf::uint32 value);

  // optional uint32 dU = 3;
  bool has_du() const;
  void clear_du();
  static const int kDUFieldNumber = 3;
  ::google::protobuf::uint32 du() const;
  void set_du(::google::protobuf::uint32 value);

  // optional uint32 dO = 4;
  bool has_do_() const;
  void clear_do_();
  static const int kDOFieldNumber = 4;
  ::google::protobuf::uint32 do_() const;
  void set_do_(::google::protobuf::uint32 value);

  // repeated float X = 5 [packed = true];
  int x_size() const;
  void clear_x();
  static const int kXFieldNumber = 5;
  float x(int index) const;
  void set_x(int index, float value);
  void add_x(float value);
  const ::google::protobuf::RepeatedField< float >&
      x() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_x();

  // repeated float U = 6 [packed = true];
  int u_size() const;
  void clear_u();
  static const int kUFieldNumber = 6;
  float u(int index) const;
  void set_u(int index, float value);
  void add_u(float value);
  const ::google::protobuf::RepeatedField< float >&
      u() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_u();

  // repeated float obs = 7 [packed = true];
  int obs_size() const;
  void clear_obs();
  static const int kObsFieldNumber = 7;
  float obs(int index) const;
  void set_obs(int index, float value);
  void add_obs(float value);
  const ::google::protobuf::RepeatedField< float >&
      obs() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_obs();

  // repeated float meta = 8 [packed = true];
  int meta_size() const;
  void clear_meta();
  static const int kMetaFieldNumber = 8;
  float meta(int index) const;
  void set_meta(int index, float value);
  void add_meta(float value);
  const ::google::protobuf::RepeatedField< float >&
      meta() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_meta();

  // @@protoc_insertion_point(class_scope:gps.Sample)
 private:
  inline void set_has_t();
  inline void clear_has_t();
  inline void set_has_dx();
  inline void clear_has_dx();
  inline void set_has_du();
  inline void clear_has_du();
  inline void set_has_do_();
  inline void clear_has_do_();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::uint32 t_;
  ::google::protobuf::uint32 dx_;
  ::google::protobuf::uint32 du_;
  ::google::protobuf::uint32 do__;
  ::google::protobuf::RepeatedField< float > x_;
  mutable int _x_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > u_;
  mutable int _u_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > obs_;
  mutable int _obs_cached_byte_size_;
  ::google::protobuf::RepeatedField< float > meta_;
  mutable int _meta_cached_byte_size_;
  friend void  protobuf_AddDesc_gps_2eproto();
  friend void protobuf_AssignDesc_gps_2eproto();
  friend void protobuf_ShutdownFile_gps_2eproto();

  void InitAsDefaultInstance();
  static Sample* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// Sample

// optional uint32 T = 1;
inline bool Sample::has_t() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Sample::set_has_t() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Sample::clear_has_t() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Sample::clear_t() {
  t_ = 0u;
  clear_has_t();
}
inline ::google::protobuf::uint32 Sample::t() const {
  // @@protoc_insertion_point(field_get:gps.Sample.T)
  return t_;
}
inline void Sample::set_t(::google::protobuf::uint32 value) {
  set_has_t();
  t_ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.T)
}

// optional uint32 dX = 2;
inline bool Sample::has_dx() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Sample::set_has_dx() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Sample::clear_has_dx() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Sample::clear_dx() {
  dx_ = 0u;
  clear_has_dx();
}
inline ::google::protobuf::uint32 Sample::dx() const {
  // @@protoc_insertion_point(field_get:gps.Sample.dX)
  return dx_;
}
inline void Sample::set_dx(::google::protobuf::uint32 value) {
  set_has_dx();
  dx_ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.dX)
}

// optional uint32 dU = 3;
inline bool Sample::has_du() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void Sample::set_has_du() {
  _has_bits_[0] |= 0x00000004u;
}
inline void Sample::clear_has_du() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void Sample::clear_du() {
  du_ = 0u;
  clear_has_du();
}
inline ::google::protobuf::uint32 Sample::du() const {
  // @@protoc_insertion_point(field_get:gps.Sample.dU)
  return du_;
}
inline void Sample::set_du(::google::protobuf::uint32 value) {
  set_has_du();
  du_ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.dU)
}

// optional uint32 dO = 4;
inline bool Sample::has_do_() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void Sample::set_has_do_() {
  _has_bits_[0] |= 0x00000008u;
}
inline void Sample::clear_has_do_() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void Sample::clear_do_() {
  do__ = 0u;
  clear_has_do_();
}
inline ::google::protobuf::uint32 Sample::do_() const {
  // @@protoc_insertion_point(field_get:gps.Sample.dO)
  return do__;
}
inline void Sample::set_do_(::google::protobuf::uint32 value) {
  set_has_do_();
  do__ = value;
  // @@protoc_insertion_point(field_set:gps.Sample.dO)
}

// repeated float X = 5 [packed = true];
inline int Sample::x_size() const {
  return x_.size();
}
inline void Sample::clear_x() {
  x_.Clear();
}
inline float Sample::x(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.X)
  return x_.Get(index);
}
inline void Sample::set_x(int index, float value) {
  x_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.X)
}
inline void Sample::add_x(float value) {
  x_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.X)
}
inline const ::google::protobuf::RepeatedField< float >&
Sample::x() const {
  // @@protoc_insertion_point(field_list:gps.Sample.X)
  return x_;
}
inline ::google::protobuf::RepeatedField< float >*
Sample::mutable_x() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.X)
  return &x_;
}

// repeated float U = 6 [packed = true];
inline int Sample::u_size() const {
  return u_.size();
}
inline void Sample::clear_u() {
  u_.Clear();
}
inline float Sample::u(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.U)
  return u_.Get(index);
}
inline void Sample::set_u(int index, float value) {
  u_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.U)
}
inline void Sample::add_u(float value) {
  u_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.U)
}
inline const ::google::protobuf::RepeatedField< float >&
Sample::u() const {
  // @@protoc_insertion_point(field_list:gps.Sample.U)
  return u_;
}
inline ::google::protobuf::RepeatedField< float >*
Sample::mutable_u() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.U)
  return &u_;
}

// repeated float obs = 7 [packed = true];
inline int Sample::obs_size() const {
  return obs_.size();
}
inline void Sample::clear_obs() {
  obs_.Clear();
}
inline float Sample::obs(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.obs)
  return obs_.Get(index);
}
inline void Sample::set_obs(int index, float value) {
  obs_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.obs)
}
inline void Sample::add_obs(float value) {
  obs_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.obs)
}
inline const ::google::protobuf::RepeatedField< float >&
Sample::obs() const {
  // @@protoc_insertion_point(field_list:gps.Sample.obs)
  return obs_;
}
inline ::google::protobuf::RepeatedField< float >*
Sample::mutable_obs() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.obs)
  return &obs_;
}

// repeated float meta = 8 [packed = true];
inline int Sample::meta_size() const {
  return meta_.size();
}
inline void Sample::clear_meta() {
  meta_.Clear();
}
inline float Sample::meta(int index) const {
  // @@protoc_insertion_point(field_get:gps.Sample.meta)
  return meta_.Get(index);
}
inline void Sample::set_meta(int index, float value) {
  meta_.Set(index, value);
  // @@protoc_insertion_point(field_set:gps.Sample.meta)
}
inline void Sample::add_meta(float value) {
  meta_.Add(value);
  // @@protoc_insertion_point(field_add:gps.Sample.meta)
}
inline const ::google::protobuf::RepeatedField< float >&
Sample::meta() const {
  // @@protoc_insertion_point(field_list:gps.Sample.meta)
  return meta_;
}
inline ::google::protobuf::RepeatedField< float >*
Sample::mutable_meta() {
  // @@protoc_insertion_point(field_mutable_list:gps.Sample.meta)
  return &meta_;
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace gps

#ifndef SWIG
namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::gps::SampleType> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::gps::SampleType>() {
  return ::gps::SampleType_descriptor();
}

}  // namespace protobuf
}  // namespace google
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_gps_2eproto__INCLUDED
