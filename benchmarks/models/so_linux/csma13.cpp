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
	uint32_t Bus;
	uint32_t S0;
	uint32_t S1;
	uint32_t S2;
	uint32_t S3;
	uint32_t S4;
	uint32_t S5;
	uint32_t S6;
	uint32_t S7;
	uint32_t S8;
	uint32_t S9;
	uint32_t S10;
	uint32_t S11;
	uint32_t S12;
	dbm::fed_t *fed;
}
__attribute__((__packed__));
int state_size = sizeof(state_struct_t);

class ClockNamesAccessor: public dbm::ClockAccessor {
public:
    virtual const std::string& getClockName(cindex_t i) const {
        switch (i) {
			case 1: { static const std::string _n("y0"); return _n; }
			case 2: { static const std::string _n("y1"); return _n; }
			case 3: { static const std::string _n("y2"); return _n; }
			case 4: { static const std::string _n("y3"); return _n; }
			case 5: { static const std::string _n("y4"); return _n; }
			case 6: { static const std::string _n("y5"); return _n; }
			case 7: { static const std::string _n("y6"); return _n; }
			case 8: { static const std::string _n("y7"); return _n; }
			case 9: { static const std::string _n("y8"); return _n; }
			case 10: { static const std::string _n("y9"); return _n; }
			case 11: { static const std::string _n("y10"); return _n; }
			case 12: { static const std::string _n("y11"); return _n; }
			case 13: { static const std::string _n("y12"); return _n; }
			case 14: { static const std::string _n("c"); return _n; }

        }
        static const std::string _ref("0");
        return _ref;
    }
};
static ClockNamesAccessor clockNames = ClockNamesAccessor();

extern "C" int get_state_variable_count() 
{
    return 16;
}

extern "C" const char* get_state_variable_name(int var)
{
    switch (var)
    {
		case 0: return "Bus";
		case 1: return "S0";
		case 2: return "S1";
		case 3: return "S2";
		case 4: return "S3";
		case 5: return "S4";
		case 6: return "S5";
		case 7: return "S6";
		case 8: return "S7";
		case 9: return "S8";
		case 10: return "S9";
		case 11: return "S10";
		case 12: return "S11";
		case 13: return "S12";
		case 14: return "fed";
		case 15: return "fed2";

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
    return 16;
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
    return 57;
}


static state_struct_t initial_state = {
	1, /* Bus */
	0, /* S0 */
	2, /* S1 */
	1, /* S2 */
	1, /* S3 */
	0, /* S4 */
	0, /* S5 */
	2, /* S6 */
	1, /* S7 */
	0, /* S8 */
	2, /* S9 */
	0, /* S10 */
	2, /* S11 */
	1, /* S12 */
	NULL, /* fed part */
};

/* Max clock constants {{{ */
int max_constants_Bus[4][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S0[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S1[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S2[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S3[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S4[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S5[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S6[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S7[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S8[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S9[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S10[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S11[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_S12[3][15] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,1,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,4,-dbm_INFINITY,},
};

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
inline void extrapolate_max_constants (state_struct_t *out) {
	int local_max_consts[15] = {0, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, };
	int i;
	for (i = 0; i < 15; i++) {
		local_max_consts[i] = max( local_max_consts[i], max_constants_Bus[out->Bus][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S0[out->S0][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S1[out->S1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S2[out->S2][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S3[out->S3][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S4[out->S4][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S5[out->S5][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S6[out->S6][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S7[out->S7][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S8[out->S8][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S9[out->S9][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S10[out->S10][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S11[out->S11][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_S12[out->S12][i] );
	}

    out->fed->extrapolateMaxBounds(local_max_consts);
}
/* }}} */

/* Invariants function {{{ */
inline bool apply_invariants (state_struct_t *out) {
	/* Invariants for Bus {{{*/
	switch( out->Bus ) {
		case 0: /* Transmitting */
		break;
		case 1: /* Idle */
		break;
		case 2: /* Active */
			/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
			if (!(true)) return false;
		break;
		case 3: /* Collided */
		break;
	}
	/* Invariants for S0 {{{*/
	switch( out->S0 ) {
		case 0: /* Wait */
		break;
		case 1: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* Retry */
		break;
	}
	/* Invariants for S1 {{{*/
	switch( out->S1 ) {
		case 0: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* Retry */
		break;
		case 2: /* Wait */
		break;
	}
	/* Invariants for S2 {{{*/
	switch( out->S2 ) {
		case 0: /* Retry */
		break;
		case 1: /* Wait */
		break;
		case 2: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(3, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for S3 {{{*/
	switch( out->S3 ) {
		case 0: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(4, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* Wait */
		break;
		case 2: /* Retry */
		break;
	}
	/* Invariants for S4 {{{*/
	switch( out->S4 ) {
		case 0: /* Wait */
		break;
		case 1: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* Retry */
		break;
	}
	/* Invariants for S5 {{{*/
	switch( out->S5 ) {
		case 0: /* Wait */
		break;
		case 1: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(6, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* Retry */
		break;
	}
	/* Invariants for S6 {{{*/
	switch( out->S6 ) {
		case 0: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(7, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* Retry */
		break;
		case 2: /* Wait */
		break;
	}
	/* Invariants for S7 {{{*/
	switch( out->S7 ) {
		case 0: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(8, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* Wait */
		break;
		case 2: /* Retry */
		break;
	}
	/* Invariants for S8 {{{*/
	switch( out->S8 ) {
		case 0: /* Wait */
		break;
		case 1: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(9, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* Retry */
		break;
	}
	/* Invariants for S9 {{{*/
	switch( out->S9 ) {
		case 0: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(10, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 1: /* Retry */
		break;
		case 2: /* Wait */
		break;
	}
	/* Invariants for S10 {{{*/
	switch( out->S10 ) {
		case 0: /* Wait */
		break;
		case 1: /* Retry */
		break;
		case 2: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(11, 0, 4, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for S11 {{{*/
	switch( out->S11 ) {
		case 0: /* Retry */
		break;
		case 1: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(12, 0, 4, false); 
			if (!(true)) return false;
		break;
		case 2: /* Wait */
		break;
	}
	/* Invariants for S12 {{{*/
	switch( out->S12 ) {
		case 0: /* Retry */
		break;
		case 1: /* Wait */
		break;
		case 2: /* Send */
			/* nop */ ; *(out->fed) &= dbm_constraint2(13, 0, 4, false); 
			if (!(true)) return false;
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
	_s.fed = new dbm::fed_t(15);
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
	/* Transitions for Bus {{{*/
	switch( in->Bus ) {
		case 0: /* Transmitting */
		break;
		case 1: /* Idle */
		break;
		case 2: /* Active */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 14, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->Bus = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 3: /* Collided */
			if( true ) {
				/* sender side of broadcast sync on cd */
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				out->Bus = 1;
				/* nop */
				/* bcast receiver Syncs with S0 on cd {{{*/
				switch( in->S0 ) {
					case 1: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S0 = 2;
							out->fed->updateValue(1, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S1 on cd {{{*/
				switch( in->S1 ) {
					case 0: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S1 = 1;
							out->fed->updateValue(2, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S2 on cd {{{*/
				switch( in->S2 ) {
					case 2: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S2 = 0;
							out->fed->updateValue(3, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S3 on cd {{{*/
				switch( in->S3 ) {
					case 0: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S3 = 2;
							out->fed->updateValue(4, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S4 on cd {{{*/
				switch( in->S4 ) {
					case 1: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S4 = 2;
							out->fed->updateValue(5, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S5 on cd {{{*/
				switch( in->S5 ) {
					case 1: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S5 = 2;
							out->fed->updateValue(6, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S6 on cd {{{*/
				switch( in->S6 ) {
					case 0: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S6 = 1;
							out->fed->updateValue(7, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S7 on cd {{{*/
				switch( in->S7 ) {
					case 0: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S7 = 2;
							out->fed->updateValue(8, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S8 on cd {{{*/
				switch( in->S8 ) {
					case 1: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S8 = 2;
							out->fed->updateValue(9, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S9 on cd {{{*/
				switch( in->S9 ) {
					case 0: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S9 = 1;
							out->fed->updateValue(10, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S10 on cd {{{*/
				switch( in->S10 ) {
					case 2: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S10 = 1;
							out->fed->updateValue(11, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S11 on cd {{{*/
				switch( in->S11 ) {
					case 1: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S11 = 0;
							out->fed->updateValue(12, 0);
						}
					break;
				}
				/* }}} */
				/* bcast receiver Syncs with S12 on cd {{{*/
				switch( in->S12 ) {
					case 2: /* Send */
						if( true ) {
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) break;
							out->S12 = 0;
							out->fed->updateValue(13, 0);
						}
					break;
				}
				/* }}} */

            system_in_deadlock = false;
            transition_info.group = -1;
            do_callback(out, &transition_info, callback, arg);
            ++states_emitted;
			}
		break;
	}
	/* }}} */
	/* Transitions for S0 {{{*/
	switch( in->S0 ) {
		case 0: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S0 = 1;
							out->Bus = 2;
							out->fed->updateValue(1, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S0 = 1;
							out->Bus = 3;
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
		case 1: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S0 = 0;
							out->Bus = 1;
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
		case 2: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 1, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S0 = 0;
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
	/* Transitions for S1 {{{*/
	switch( in->S1 ) {
		case 0: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S1 = 2;
							out->Bus = 1;
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
		case 1: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 2, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S1 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S1 = 0;
							out->Bus = 2;
							out->fed->updateValue(2, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S1 = 0;
							out->Bus = 3;
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
	/* Transitions for S2 {{{*/
	switch( in->S2 ) {
		case 0: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 3, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S2 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S2 = 2;
							out->Bus = 2;
							out->fed->updateValue(3, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S2 = 2;
							out->Bus = 3;
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
		case 2: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S2 = 1;
							out->Bus = 1;
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
	/* Transitions for S3 {{{*/
	switch( in->S3 ) {
		case 0: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S3 = 1;
							out->Bus = 1;
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
		case 1: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S3 = 0;
							out->Bus = 2;
							out->fed->updateValue(4, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S3 = 0;
							out->Bus = 3;
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
		case 2: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 4, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S3 = 1;
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
	/* Transitions for S4 {{{*/
	switch( in->S4 ) {
		case 0: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S4 = 1;
							out->Bus = 2;
							out->fed->updateValue(5, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S4 = 1;
							out->Bus = 3;
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
		case 1: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S4 = 0;
							out->Bus = 1;
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
		case 2: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 5, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S4 = 0;
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
	/* Transitions for S5 {{{*/
	switch( in->S5 ) {
		case 0: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S5 = 1;
							out->Bus = 2;
							out->fed->updateValue(6, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S5 = 1;
							out->Bus = 3;
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
		case 1: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S5 = 0;
							out->Bus = 1;
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
		case 2: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 6, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S5 = 0;
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
	/* Transitions for S6 {{{*/
	switch( in->S6 ) {
		case 0: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S6 = 2;
							out->Bus = 1;
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
		case 1: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 7, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S6 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S6 = 0;
							out->Bus = 2;
							out->fed->updateValue(7, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S6 = 0;
							out->Bus = 3;
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
	/* Transitions for S7 {{{*/
	switch( in->S7 ) {
		case 0: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S7 = 1;
							out->Bus = 1;
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
		case 1: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S7 = 0;
							out->Bus = 2;
							out->fed->updateValue(8, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S7 = 0;
							out->Bus = 3;
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
		case 2: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 8, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S7 = 1;
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
	/* Transitions for S8 {{{*/
	switch( in->S8 ) {
		case 0: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S8 = 1;
							out->Bus = 2;
							out->fed->updateValue(9, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S8 = 1;
							out->Bus = 3;
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
		case 1: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S8 = 0;
							out->Bus = 1;
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
		case 2: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 9, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S8 = 0;
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
	/* Transitions for S9 {{{*/
	switch( in->S9 ) {
		case 0: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S9 = 2;
							out->Bus = 1;
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
		case 1: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 10, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S9 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S9 = 0;
							out->Bus = 2;
							out->fed->updateValue(10, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S9 = 0;
							out->Bus = 3;
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
	/* Transitions for S10 {{{*/
	switch( in->S10 ) {
		case 0: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S10 = 2;
							out->Bus = 2;
							out->fed->updateValue(11, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S10 = 2;
							out->Bus = 3;
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
		case 1: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 11, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S10 = 0;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S10 = 0;
							out->Bus = 1;
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
	/* Transitions for S11 {{{*/
	switch( in->S11 ) {
		case 0: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 12, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S11 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 12, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S11 = 2;
							out->Bus = 1;
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
		case 2: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S11 = 1;
							out->Bus = 2;
							out->fed->updateValue(12, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S11 = 1;
							out->Bus = 3;
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
	/* Transitions for S12 {{{*/
	switch( in->S12 ) {
		case 0: /* Retry */
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 13, -1, false); 
				if (!(out->fed->isEmpty())) {
				out->S12 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* Wait */
			if( true ) {
				/* Syncs with Bus on begin {{{*/
				switch( in->Bus ) {
					case 1: /* Idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S12 = 2;
							out->Bus = 2;
							out->fed->updateValue(13, 0);
							out->fed->updateValue(14, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2: /* Active */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; *(out->fed) &= dbm_constraint2(14, 0, 1, false); 
							if (!(out->fed->isEmpty())) {
							out->S12 = 2;
							out->Bus = 3;
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
		case 2: /* Send */
			if( true ) {
				/* Syncs with Bus on end {{{*/
				switch( in->Bus ) {
					case 0: /* Transmitting */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 13, -4, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->S12 = 1;
							out->Bus = 1;
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
	if (states_emitted) return states_emitted;
	return states_emitted;
};


 
extern "C" int get_successor( void* model, int t, state_struct_t *in, void (*callback)(void* arg, transition_info_t *transition_info, state_struct_t *out), void *arg ) 
{
    return 0;
}

int transition_dependency[16] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, };

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
