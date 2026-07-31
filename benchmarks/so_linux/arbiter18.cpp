#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <map>
#include <mutex>
#include <string>

typedef uint64_t ulong_long_int_t;
typedef int64_t slong_long_int_t;
typedef uint32_t ulong_int_t;
typedef int32_t slong_int_t;
typedef union
{
    uint16_t var;
    uint32_t __padding__;
} ushort_int_t;
typedef union
{
    int16_t var;
    uint32_t __padding__;
} sshort_int_t;
typedef union
{
    uint8_t var;
    uint32_t __padding__;
} ubyte_t;
typedef ubyte_t byte_t;
typedef union
{
    int8_t var;
    uint32_t __padding__;
} sbyte_t;
typedef union
{
    uint8_t var;
    uint32_t __padding__;
} bool_t;
typedef size_t size_int_t;

typedef struct transition_info
{
    int* label;
    int  group;
} transition_info_t;

#include "dbm/fed.h"
#include "dbm/ClockAccessor.h"


struct state_struct_t
{
	uint32_t Arb;
	uint32_t C0;
	uint32_t C1;
	uint32_t C2;
	uint32_t C3;
	uint32_t C4;
	uint32_t C5;
	uint32_t C6;
	uint32_t C7;
	uint32_t C8;
	uint32_t C9;
	uint32_t C10;
	uint32_t C11;
	uint32_t C12;
	uint32_t C13;
	uint32_t C14;
	uint32_t C15;
	uint32_t C16;
	uint32_t C17;
	dbm::fed_t *fed;
}
__attribute__((__packed__));
int state_size = sizeof(state_struct_t);

class ClockNamesAccessor: public dbm::ClockAccessor {
public:
    virtual const std::string& getClockName(cindex_t i) const {
        switch (i) {
			case 1: { static const std::string _n("x0"); return _n; }
			case 2: { static const std::string _n("x1"); return _n; }
			case 3: { static const std::string _n("x2"); return _n; }
			case 4: { static const std::string _n("x3"); return _n; }
			case 5: { static const std::string _n("x4"); return _n; }
			case 6: { static const std::string _n("x5"); return _n; }
			case 7: { static const std::string _n("x6"); return _n; }
			case 8: { static const std::string _n("x7"); return _n; }
			case 9: { static const std::string _n("x8"); return _n; }
			case 10: { static const std::string _n("x9"); return _n; }
			case 11: { static const std::string _n("x10"); return _n; }
			case 12: { static const std::string _n("x11"); return _n; }
			case 13: { static const std::string _n("x12"); return _n; }
			case 14: { static const std::string _n("x13"); return _n; }
			case 15: { static const std::string _n("x14"); return _n; }
			case 16: { static const std::string _n("x15"); return _n; }
			case 17: { static const std::string _n("x16"); return _n; }
			case 18: { static const std::string _n("x17"); return _n; }

        }
        static const std::string _ref("0");
        return _ref;
    }
};
static ClockNamesAccessor clockNames = ClockNamesAccessor();

extern "C" int get_state_variable_count() 
{
    return 21;
}

extern "C" const char* get_state_variable_name(int var)
{
    switch (var)
    {
		case 0: return "Arb";
		case 1: return "C0";
		case 2: return "C1";
		case 3: return "C2";
		case 4: return "C3";
		case 5: return "C4";
		case 6: return "C5";
		case 7: return "C6";
		case 8: return "C7";
		case 9: return "C8";
		case 10: return "C9";
		case 11: return "C10";
		case 12: return "C11";
		case 13: return "C12";
		case 14: return "C13";
		case 15: return "C14";
		case 16: return "C15";
		case 17: return "C16";
		case 18: return "C17";
		case 19: return "fed";
		case 20: return "fed2";

        default:
            return NULL;
    }
}

extern "C" int get_state_variable_type(int var)
{
    return 0;
}
extern "C" int get_state_variable_type_count() 
{
    return 1;
}
extern "C" const char* get_state_variable_type_name(int type) 
{
    return "int";
}

extern "C" int get_state_variable_type_value_count(int type)
{
    return 21;
}

extern "C" const char* get_state_variable_type_value(int type, int value)
{
    /* Must be a *distinct* string per value: LTSmin interns these in a
       lock-less chunk set (mc-lib/set-ll.c) that requires a bijection between
       index and string. Returning one constant for every value trips
       "isba_size_int(balloc) == index + 1" as soon as >1 worker installs
       chunks concurrently.

       It must also be STABLE for the lifetime of the process. LTSmin retains
       the pointer and stores a 16-bit hash tag beside the key; when the table
       grows, hti_copy_entry re-validates
           (key_hash >> 16) == (ht1_ent->key >> 48)
       against the string's *current* contents (mc-lib/hashtable.c:274). A
       reused per-thread scratch buffer mutates the already-interned string and
       trips that assertion -- observed on viking17 at >=4 threads, which is the
       first model with enough distinct chunk values to force a resize.
       So: allocate once per value, never rewrite, never free. std::map keeps
       element addresses stable across later insertions. */
    static std::mutex chunk_mutex;
    static std::map<int, std::string> chunk_cache;
    std::lock_guard<std::mutex> lock(chunk_mutex);
    std::map<int, std::string>::iterator it = chunk_cache.find(value);
    if (it == chunk_cache.end()) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "avar%d", value);
        it = chunk_cache.insert(std::make_pair(value, std::string(tmp))).first;
    }
    return it->second.c_str();
}

extern "C" int get_transition_count() 
{
    return 56;
}


static state_struct_t initial_state = {
	1, /* Arb */
	2, /* C0 */
	2, /* C1 */
	0, /* C2 */
	1, /* C3 */
	0, /* C4 */
	1, /* C5 */
	2, /* C6 */
	0, /* C7 */
	2, /* C8 */
	2, /* C9 */
	0, /* C10 */
	1, /* C11 */
	0, /* C12 */
	1, /* C13 */
	2, /* C14 */
	2, /* C15 */
	2, /* C16 */
	0, /* C17 */
	NULL, /* fed part */
};

/* Max clock constants {{{ */
int max_constants_Arb[2][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C0[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C1[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C2[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C3[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C4[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C5[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C6[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C7[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C8[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C9[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C10[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C11[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C12[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C13[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C14[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C15[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_C16[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,},
};
int max_constants_C17[3][19] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
inline void extrapolate_max_constants (state_struct_t *out) {
	int local_max_consts[19] = {0, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, };
	int i;
	for (i = 0; i < 19; i++) {
		local_max_consts[i] = max( local_max_consts[i], max_constants_Arb[out->Arb][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C0[out->C0][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C1[out->C1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C2[out->C2][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C3[out->C3][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C4[out->C4][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C5[out->C5][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C6[out->C6][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C7[out->C7][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C8[out->C8][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C9[out->C9][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C10[out->C10][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C11[out->C11][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C12[out->C12][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C13[out->C13][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C14[out->C14][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C15[out->C15][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C16[out->C16][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_C17[out->C17][i] );
	}

    out->fed->extrapolateMaxBounds(local_max_consts);
}
/* }}} */

/* Invariants function {{{ */
inline bool apply_invariants (state_struct_t *out) {
	/* Invariants for C0 {{{*/
	switch( out->C0 ) {
		case 0: /* want */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C1 {{{*/
	switch( out->C1 ) {
		case 0: /* want */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C2 {{{*/
	switch( out->C2 ) {
		case 0: /* idle */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(3, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* want */
		break;
	}
	/* Invariants for C3 {{{*/
	switch( out->C3 ) {
		case 0: /* want */
		break;
		case 1: /* idle */
		break;
		case 2: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(4, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for C4 {{{*/
	switch( out->C4 ) {
		case 0: /* idle */
		break;
		case 1: /* want */
		break;
		case 2: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for C5 {{{*/
	switch( out->C5 ) {
		case 0: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(6, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* idle */
		break;
		case 2: /* want */
		break;
	}
	/* Invariants for C6 {{{*/
	switch( out->C6 ) {
		case 0: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(7, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* want */
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C7 {{{*/
	switch( out->C7 ) {
		case 0: /* idle */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(8, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* want */
		break;
	}
	/* Invariants for C8 {{{*/
	switch( out->C8 ) {
		case 0: /* want */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(9, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C9 {{{*/
	switch( out->C9 ) {
		case 0: /* want */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(10, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C10 {{{*/
	switch( out->C10 ) {
		case 0: /* idle */
		break;
		case 1: /* want */
		break;
		case 2: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(11, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for C11 {{{*/
	switch( out->C11 ) {
		case 0: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(12, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* idle */
		break;
		case 2: /* want */
		break;
	}
	/* Invariants for C12 {{{*/
	switch( out->C12 ) {
		case 0: /* idle */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* want */
		break;
	}
	/* Invariants for C13 {{{*/
	switch( out->C13 ) {
		case 0: /* want */
		break;
		case 1: /* idle */
		break;
		case 2: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for C14 {{{*/
	switch( out->C14 ) {
		case 0: /* want */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(15, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C15 {{{*/
	switch( out->C15 ) {
		case 0: /* want */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(16, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C16 {{{*/
	switch( out->C16 ) {
		case 0: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(17, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* want */
		break;
		case 2: /* idle */
		break;
	}
	/* Invariants for C17 {{{*/
	switch( out->C17 ) {
		case 0: /* idle */
		break;
		case 1: /* cs */
			/* nop */ ; *(out->fed) &= dbm_constraint2(18, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* want */
		break;
	}
	if (out->fed->isEmpty()) return false;
	return true;
}
/* }}} */


extern "C" int get_state_size() {
    return state_size;
}

extern "C" void get_initial_state( void *to )
{
    /* Every LTSmin worker calls this. The original built the state in the
       shared `initial_state` global, so concurrent workers raced on it (and
       leaked a federation per call). Build in a thread-local copy instead. */
    state_struct_t _s = initial_state;
	_s.fed = new dbm::fed_t(19);
	_s.fed->setZero();
	_s.fed->up();
	assert(apply_invariants(&_s));
	extrapolate_max_constants(&_s);

    memcpy(to, &_s, state_size);
}

extern "C" int have_property()
{
    return false;
}


void do_callback(state_struct_t *out, transition_info_t *transition_info, void (*callback)(void *arg, transition_info_t *transition_info, state_struct_t *out), void *arg) {
	out->fed->up();
	if (apply_invariants(out)) {
		extrapolate_max_constants(out);
		out->fed->mergeReduce();

                callback(arg, transition_info, out);
        }
}


extern "C" int get_successors( void *model, state_struct_t *in, void (*callback)(void *arg, transition_info_t *transition_info, state_struct_t *out), void *arg ) 
{
    (void)model; // ignore model
    bool system_in_deadlock = true;
    transition_info_t transition_info = { NULL, -1 };
    int states_emitted = 0;
    state_struct_t tmp;
    state_struct_t *out = &tmp;
	/* Transitions for Arb {{{*/
	switch( in->Arb ) {
		case 0: /* Busy */
		break;
		case 1: /* Free */
		break;
	}
	/* }}} */
	/* Transitions for C0 {{{*/
	switch( in->C0 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C0 = 1;
							out->Arb = 0;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C0 = 2;
							out->Arb = 1;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 1, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C0 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C1 {{{*/
	switch( in->C1 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C1 = 1;
							out->Arb = 0;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C1 = 2;
							out->Arb = 1;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 2, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C1 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C2 {{{*/
	switch( in->C2 ) {
		case 0: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 3, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C2 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C2 = 0;
							out->Arb = 1;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C2 = 1;
							out->Arb = 0;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C3 {{{*/
	switch( in->C3 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C3 = 2;
							out->Arb = 0;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 4, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C3 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C3 = 1;
							out->Arb = 1;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C4 {{{*/
	switch( in->C4 ) {
		case 0: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 5, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C4 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C4 = 2;
							out->Arb = 0;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C4 = 0;
							out->Arb = 1;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C5 {{{*/
	switch( in->C5 ) {
		case 0: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C5 = 1;
							out->Arb = 1;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 6, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C5 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C5 = 0;
							out->Arb = 0;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C6 {{{*/
	switch( in->C6 ) {
		case 0: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C6 = 2;
							out->Arb = 1;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C6 = 0;
							out->Arb = 0;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 7, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C6 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C7 {{{*/
	switch( in->C7 ) {
		case 0: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 8, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C7 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C7 = 0;
							out->Arb = 1;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C7 = 1;
							out->Arb = 0;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C8 {{{*/
	switch( in->C8 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C8 = 1;
							out->Arb = 0;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C8 = 2;
							out->Arb = 1;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 9, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C8 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C9 {{{*/
	switch( in->C9 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C9 = 1;
							out->Arb = 0;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C9 = 2;
							out->Arb = 1;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 10, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C9 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C10 {{{*/
	switch( in->C10 ) {
		case 0: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 11, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C10 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C10 = 2;
							out->Arb = 0;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C10 = 0;
							out->Arb = 1;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C11 {{{*/
	switch( in->C11 ) {
		case 0: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C11 = 1;
							out->Arb = 1;
							out->fed->updateValue(12, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 12, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C11 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C11 = 0;
							out->Arb = 0;
							out->fed->updateValue(12, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C12 {{{*/
	switch( in->C12 ) {
		case 0: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 13, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C12 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C12 = 0;
							out->Arb = 1;
							out->fed->updateValue(13, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C12 = 1;
							out->Arb = 0;
							out->fed->updateValue(13, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C13 {{{*/
	switch( in->C13 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C13 = 2;
							out->Arb = 0;
							out->fed->updateValue(14, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 14, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C13 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C13 = 1;
							out->Arb = 1;
							out->fed->updateValue(14, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	/* Transitions for C14 {{{*/
	switch( in->C14 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C14 = 1;
							out->Arb = 0;
							out->fed->updateValue(15, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C14 = 2;
							out->Arb = 1;
							out->fed->updateValue(15, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 15, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C14 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C15 {{{*/
	switch( in->C15 ) {
		case 0: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C15 = 1;
							out->Arb = 0;
							out->fed->updateValue(16, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C15 = 2;
							out->Arb = 1;
							out->fed->updateValue(16, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 16, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C15 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C16 {{{*/
	switch( in->C16 ) {
		case 0: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C16 = 2;
							out->Arb = 1;
							out->fed->updateValue(17, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 1: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C16 = 0;
							out->Arb = 0;
							out->fed->updateValue(17, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 17, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C16 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for C17 {{{*/
	switch( in->C17 ) {
		case 0: /* idle */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 18, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->C17 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* cs */
			if( true ) {
				/* Syncs with Arb on rel {{{*/
				switch( in->Arb ) {
					case 0: /* Busy */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C17 = 0;
							out->Arb = 1;
							out->fed->updateValue(18, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
		case 2: /* want */
			if( true ) {
				/* Syncs with Arb on acq {{{*/
				switch( in->Arb ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->C17 = 1;
							out->Arb = 0;
							out->fed->updateValue(18, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
				}
				/* }}} */
			}
		break;
	}
	/* }}} */
	if (states_emitted) return states_emitted;
	return states_emitted;
};


 
extern "C" int get_successor( void* model, int t, state_struct_t *in, void (*callback)(void* arg, transition_info_t *transition_info, state_struct_t *out), void *arg ) 
{
    return 0;
}

int transition_dependency[21] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, };

extern "C" const int* get_transition_read_dependencies(int t) 
{
    return transition_dependency;
}
extern "C" const int* get_transition_write_dependencies(int t) 
{
    return transition_dependency;
}

extern "C" const int get_guard_count() 
{
    return 0;
}
extern "C" const int* get_guard_matrix(int g) 
{
    return NULL;
}
extern "C" const int* get_guards(int t) 
{
    return NULL;
}
extern "C" const int** get_all_guards() 
{
    return NULL;
}
extern "C" int get_guard(void* model, int g, state_struct_t* src) 
{
    return 0;
}
extern "C" void get_guard_all(void* model, state_struct_t* src, int* guard) 
{
}
extern "C" const int* get_guard_may_be_coenabled_matrix(int g) 
{
    return NULL;
}
extern "C" const int* get_guard_nes_matrix(int g) 
{
    return NULL;
}
extern "C" const int* get_guard_nds_matrix(int g) 
{
    return NULL;
}


typedef union fed_u {
   dbm::fed_t  *fed;
   uint64_t    fill;
} fed_u_t;

extern "C" int covered_by_short( int *fed_a, int *fed_b) 
{
    fed_u_t pa = *((fed_u_t*)fed_a);
    fed_u_t pb = *((fed_u_t*)fed_b);

    /*std::cout << "cover: " << 
	    pa.fed->str(clockNames) << 
	    " " << 
	    pb.fed->str(clockNames) << std::endl;*/

    if ( *(pa.fed) <= *(pb.fed) )
	return 1;
    else
        return 0;
}

/* LTSmin >= 2.x lattice interface. This opaal revision predates it, so the
   five lattice_* hooks are supplied here. The `lattice` pointer LTSmin hands
   us is exactly the dbm::fed_t* stored in the last two slots of the state
   vector. LTSmin's hashtable treats cmp()==0 as "equal" (mc-lib/hashtable.c). */
extern "C" void* lattice_clone (const void *lattice)
{
    return new dbm::fed_t(*((const dbm::fed_t*)lattice));
}

extern "C" int lattice_cmp (const void *l1, const void *l2)
{
    return (*((const dbm::fed_t*)l1) == *((const dbm::fed_t*)l2)) ? 0 : 1;
}

extern "C" uint32_t lattice_hash (const void *lattice)
{
    return ((const dbm::fed_t*)lattice)->hash();
}

extern "C" void lattice_delete (const void *lattice)
{
    delete ((const dbm::fed_t*)lattice);
}

extern "C" const char* lattice_print (const void *lattice)
{
    /* LTSmin only reads the result before the next call on this thread. */
    static __thread std::string *buf = NULL;
    if (buf == NULL) buf = new std::string();
    *buf = ((const dbm::fed_t*)lattice)->str(clockNames);
    return buf->c_str();
}

void dummy_cb(void *arg, transition_info_t *transition_info, state_struct_t *out) {
	printf("succ_cb: %p\n", out->fed);
	std::cout << "succ_cb: " << out->fed->str(clockNames) << std::endl;
}

int main() {
    //g++ -g -shared -O0 -fPIC -I/usr/local/uppaal/include/ -L/usr/local/uppaal/lib/ -o gensuccgen gensuccgen.cpp -ludbm
    state_struct_t *s1 = (state_struct_t*)malloc(get_state_size());
    get_initial_state(s1);
	std::cout << "initial: " << s1->fed->str(clockNames) << std::endl;

	get_successors(NULL, s1, dummy_cb, NULL);

    return 0;
}
/* vim:ts=4:sw=4:noexpandtab */
