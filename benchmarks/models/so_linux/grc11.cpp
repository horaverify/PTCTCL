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
	uint32_t T0;
	uint32_t T1;
	uint32_t T2;
	uint32_t T3;
	uint32_t T4;
	uint32_t T5;
	uint32_t T6;
	uint32_t T7;
	uint32_t T8;
	uint32_t T9;
	uint32_t T10;
	uint32_t G;
	uint32_t Ctl;
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
			case 12: { static const std::string _n("y"); return _n; }
			case 13: { static const std::string _n("z"); return _n; }

        }
        static const std::string _ref("0");
        return _ref;
    }
};
static ClockNamesAccessor clockNames = ClockNamesAccessor();

extern "C" int get_state_variable_count() 
{
    return 15;
}

extern "C" const char* get_state_variable_name(int var)
{
    switch (var)
    {
		case 0: return "T0";
		case 1: return "T1";
		case 2: return "T2";
		case 3: return "T3";
		case 4: return "T4";
		case 5: return "T5";
		case 6: return "T6";
		case 7: return "T7";
		case 8: return "T8";
		case 9: return "T9";
		case 10: return "T10";
		case 11: return "G";
		case 12: return "Ctl";
		case 13: return "fed";
		case 14: return "fed2";

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
    return 15;
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
    return 82;
}


static state_struct_t initial_state = {
	0, /* T0 */
	2, /* T1 */
	1, /* T2 */
	2, /* T3 */
	1, /* T4 */
	0, /* T5 */
	2, /* T6 */
	1, /* T7 */
	0, /* T8 */
	2, /* T9 */
	2, /* T10 */
	3, /* G */
	18, /* Ctl */
	NULL, /* fed part */
};

/* Max clock constants {{{ */
int max_constants_T0[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T1[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T2[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T3[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T4[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T5[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T6[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T7[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T8[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T9[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_T10[3][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,15,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_G[4][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Ctl[24][14] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
inline void extrapolate_max_constants (state_struct_t *out) {
	int local_max_consts[14] = {0, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, };
	int i;
	for (i = 0; i < 14; i++) {
		local_max_consts[i] = max( local_max_consts[i], max_constants_T0[out->T0][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T1[out->T1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T2[out->T2][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T3[out->T3][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T4[out->T4][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T5[out->T5][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T6[out->T6][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T7[out->T7][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T8[out->T8][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T9[out->T9][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_T10[out->T10][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_G[out->G][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Ctl[out->Ctl][i] );
	}

    out->fed->extrapolateMaxBounds(local_max_consts);
}
/* }}} */

/* Invariants function {{{ */
inline bool apply_invariants (state_struct_t *out) {
	/* Invariants for T0 {{{*/
	switch( out->T0 ) {
		case 0: /* far */
		break;
		case 1: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 2: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for T1 {{{*/
	switch( out->T1 ) {
		case 0: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 1: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* far */
		break;
	}
	/* Invariants for T2 {{{*/
	switch( out->T2 ) {
		case 0: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(3, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 1: /* far */
		break;
		case 2: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(3, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for T3 {{{*/
	switch( out->T3 ) {
		case 0: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(4, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(4, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 2: /* far */
		break;
	}
	/* Invariants for T4 {{{*/
	switch( out->T4 ) {
		case 0: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* far */
		break;
		case 2: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, 15, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for T5 {{{*/
	switch( out->T5 ) {
		case 0: /* far */
		break;
		case 1: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(6, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 2: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(6, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for T6 {{{*/
	switch( out->T6 ) {
		case 0: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(7, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(7, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 2: /* far */
		break;
	}
	/* Invariants for T7 {{{*/
	switch( out->T7 ) {
		case 0: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(8, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 1: /* far */
		break;
		case 2: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(8, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for T8 {{{*/
	switch( out->T8 ) {
		case 0: /* far */
		break;
		case 1: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(9, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 2: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(9, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for T9 {{{*/
	switch( out->T9 ) {
		case 0: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(10, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 1: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(10, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* far */
		break;
	}
	/* Invariants for T10 {{{*/
	switch( out->T10 ) {
		case 0: /* near */
			/* nop */ ; *(out->fed) &= dbm_constraint2(11, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* in */
			/* nop */ ; *(out->fed) &= dbm_constraint2(11, 0, 15, false); 
			if (!(true)) return false;
		break;
		case 2: /* far */
		break;
	}
	/* Invariants for G {{{*/
	switch( out->G ) {
		case 0: /* low */
			/* nop */ ; *(out->fed) &= dbm_constraint2(12, 0, 2, false); 
			if (!(true)) return false;
		break;
		case 1: /* rai */
			/* nop */ ; *(out->fed) &= dbm_constraint2(12, 0, 2, false); 
			if (!(true)) return false;
		break;
		case 2: /* down */
		break;
		case 3: /* up */
		break;
	}
	/* Invariants for Ctl {{{*/
	switch( out->Ctl ) {
		case 0: /* dn_1 */
		break;
		case 1: /* dn_3 */
		break;
		case 2: /* dn_2 */
		break;
		case 3: /* dn_5 */
		break;
		case 4: /* dn_4 */
		break;
		case 5: /* dn_7 */
		break;
		case 6: /* dn_6 */
		break;
		case 7: /* dn_9 */
		break;
		case 8: /* dn_8 */
		break;
		case 9: /* low_8 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 10: /* low_9 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 11: /* low_6 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 12: /* low_7 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 13: /* low_4 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 14: /* low_5 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 15: /* low_2 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 16: /* low_3 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 17: /* low_1 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 18: /* up */
		break;
		case 19: /* rai */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 20: /* low_10 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 21: /* low_11 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 22: /* dn_11 */
		break;
		case 23: /* dn_10 */
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
	_s.fed = new dbm::fed_t(14);
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
	/* Transitions for T0 {{{*/
	switch( in->T0 ) {
		case 0: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 2;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 4;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 1;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 6;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 3;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 8;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 5;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 23;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 7;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 10;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 20;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 12;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 9;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 14;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 11;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 16;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 13;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 15;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 17;
							out->fed->updateValue(1, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 21;
							out->fed->updateValue(1, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 2;
							out->Ctl = 22;
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
		case 1: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T0 = 0;
							out->Ctl = 7;
							/* nop */
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
		case 2: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 1, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T0 = 1;
				out->fed->updateValue(1, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for T1 {{{*/
	switch( in->T1 ) {
		case 0: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 2;
							out->Ctl = 7;
							/* nop */
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
		case 1: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 2, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T1 = 0;
				out->fed->updateValue(2, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 2;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 4;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 1;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 6;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 3;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 8;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 5;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 23;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 7;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 10;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 20;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 12;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 9;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 14;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 11;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 16;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 13;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 15;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 17;
							out->fed->updateValue(2, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 21;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T1 = 1;
							out->Ctl = 22;
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
	}
	/* }}} */
	/* Transitions for T2 {{{*/
	switch( in->T2 ) {
		case 0: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 1;
							out->Ctl = 7;
							/* nop */
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
		case 1: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 2;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 4;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 1;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 6;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 3;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 8;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 5;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 23;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 7;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 10;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 20;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 12;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 9;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 14;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 11;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 16;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 13;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 15;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 17;
							out->fed->updateValue(3, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 21;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T2 = 2;
							out->Ctl = 22;
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
		case 2: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 3, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T2 = 0;
				out->fed->updateValue(3, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for T3 {{{*/
	switch( in->T3 ) {
		case 0: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 4, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T3 = 1;
				out->fed->updateValue(4, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 2;
							out->Ctl = 7;
							/* nop */
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
		case 2: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 2;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 4;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 1;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 6;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 3;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 8;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 5;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 23;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 7;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 10;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 20;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 12;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 9;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 14;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 11;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 16;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 13;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 15;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 17;
							out->fed->updateValue(4, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 21;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T3 = 0;
							out->Ctl = 22;
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
	/* Transitions for T4 {{{*/
	switch( in->T4 ) {
		case 0: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 5, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T4 = 2;
				out->fed->updateValue(5, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 2;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 4;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 1;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 6;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 3;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 8;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 5;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 23;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 7;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 10;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 20;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 12;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 9;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 14;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 11;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 16;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 13;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 15;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 17;
							out->fed->updateValue(5, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 21;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 0;
							out->Ctl = 22;
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
		case 2: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T4 = 1;
							out->Ctl = 7;
							/* nop */
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
	/* Transitions for T5 {{{*/
	switch( in->T5 ) {
		case 0: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 2;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 4;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 1;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 6;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 3;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 8;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 5;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 23;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 7;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 10;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 20;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 12;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 9;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 14;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 11;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 16;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 13;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 15;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 17;
							out->fed->updateValue(6, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 21;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 2;
							out->Ctl = 22;
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
		case 1: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T5 = 0;
							out->Ctl = 7;
							/* nop */
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
		case 2: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 6, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T5 = 1;
				out->fed->updateValue(6, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for T6 {{{*/
	switch( in->T6 ) {
		case 0: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 7, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T6 = 1;
				out->fed->updateValue(7, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 2;
							out->Ctl = 7;
							/* nop */
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
		case 2: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 2;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 4;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 1;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 6;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 3;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 8;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 5;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 23;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 7;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 10;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 20;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 12;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 9;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 14;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 11;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 16;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 13;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 15;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 17;
							out->fed->updateValue(7, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 21;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T6 = 0;
							out->Ctl = 22;
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
	}
	/* }}} */
	/* Transitions for T7 {{{*/
	switch( in->T7 ) {
		case 0: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 1;
							out->Ctl = 7;
							/* nop */
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
		case 1: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 2;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 4;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 1;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 6;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 3;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 8;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 5;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 23;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 7;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 10;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 20;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 12;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 9;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 14;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 11;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 16;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 13;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 15;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 17;
							out->fed->updateValue(8, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 21;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T7 = 2;
							out->Ctl = 22;
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
		case 2: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 8, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T7 = 0;
				out->fed->updateValue(8, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for T8 {{{*/
	switch( in->T8 ) {
		case 0: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 2;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 4;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 1;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 6;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 3;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 8;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 5;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 23;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 7;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 10;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 20;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 12;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 9;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 14;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 11;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 16;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 13;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 15;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 17;
							out->fed->updateValue(9, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 21;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 2;
							out->Ctl = 22;
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
		case 1: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T8 = 0;
							out->Ctl = 7;
							/* nop */
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
		case 2: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 9, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T8 = 1;
				out->fed->updateValue(9, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
	}
	/* }}} */
	/* Transitions for T9 {{{*/
	switch( in->T9 ) {
		case 0: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 2;
							out->Ctl = 7;
							/* nop */
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
		case 1: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 10, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T9 = 0;
				out->fed->updateValue(10, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 2;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 4;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 1;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 6;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 3;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 8;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 5;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 23;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 7;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 10;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 20;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 12;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 9;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 14;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 11;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 16;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 13;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 15;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 17;
							out->fed->updateValue(10, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 21;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T9 = 1;
							out->Ctl = 22;
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
	}
	/* }}} */
	/* Transitions for T10 {{{*/
	switch( in->T10 ) {
		case 0: /* near */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 11, -4, false); 
				if (!(out->fed->isEmpty())) {
				out->T10 = 1;
				out->fed->updateValue(11, 0);

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* in */
			if( true ) {
				/* Syncs with Ctl on leave {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 19;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 2;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 0;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 4;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 1;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 6;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 8;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 5;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 18;
							/* nop */
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 22: /* dn_11 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 23;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 2;
							out->Ctl = 7;
							/* nop */
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
		case 2: /* far */
			if( true ) {
				/* Syncs with Ctl on approach {{{*/
				switch( in->Ctl ) {
					case 0: /* dn_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 2;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 1: /* dn_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 4;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* dn_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 1;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* dn_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 6;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* dn_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 3;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 5: /* dn_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 8;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 6: /* dn_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 5;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 7: /* dn_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 23;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 8: /* dn_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 7;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 9: /* low_8 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 10;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 10: /* low_9 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 20;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 11: /* low_6 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 12;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 12: /* low_7 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 9;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 13: /* low_4 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 14;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 14: /* low_5 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 11;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 15: /* low_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 16;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 16: /* low_3 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 13;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 17: /* low_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 15;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 18: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 17;
							out->fed->updateValue(11, 0);
							out->fed->updateValue(13, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 20: /* low_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 21;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 23: /* dn_10 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->T10 = 0;
							out->Ctl = 22;
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
	/* Transitions for G {{{*/
	switch( in->G ) {
		case 0: /* low */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 12, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->G = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* rai */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 12, -2, false); 
				if (!(out->fed->isEmpty())) {
				out->G = 3;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* down */
		break;
		case 3: /* up */
		break;
	}
	/* }}} */
	/* Transitions for Ctl {{{*/
	switch( in->Ctl ) {
		case 0: /* dn_1 */
		break;
		case 1: /* dn_3 */
		break;
		case 2: /* dn_2 */
		break;
		case 3: /* dn_5 */
		break;
		case 4: /* dn_4 */
		break;
		case 5: /* dn_7 */
		break;
		case 6: /* dn_6 */
		break;
		case 7: /* dn_9 */
		break;
		case 8: /* dn_8 */
		break;
		case 9: /* low_8 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 8;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 10: /* low_9 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 7;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 11: /* low_6 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 6;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 12: /* low_7 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 5;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 13: /* low_4 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 4;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 14: /* low_5 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 3;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 15: /* low_2 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 2;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 16: /* low_3 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 1;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 17: /* low_1 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 0;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 18: /* up */
		break;
		case 19: /* rai */
			if( true ) {
				/* Syncs with G on raise {{{*/
				switch( in->G ) {
					case 2: /* down */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 18;
							out->G = 1;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 20: /* low_10 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 23;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 21: /* low_11 */
			if( true ) {
				/* Syncs with G on lower {{{*/
				switch( in->G ) {
					case 3: /* up */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Ctl = 22;
							out->G = 0;
							/* nop */
							out->fed->updateValue(12, 0);

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
		case 22: /* dn_11 */
		break;
		case 23: /* dn_10 */
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

int transition_dependency[15] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, };

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
