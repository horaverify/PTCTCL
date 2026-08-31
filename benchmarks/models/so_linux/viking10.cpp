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
	uint32_t L;
	uint32_t Viking1;
	uint32_t Viking2;
	uint32_t Viking3;
	uint32_t Viking4;
	uint32_t Viking5;
	uint32_t Viking6;
	uint32_t Viking7;
	uint32_t Viking8;
	uint32_t Viking9;
	uint32_t Viking10;
	uint32_t Torch;
	dbm::fed_t *fed;
}
__attribute__((__packed__));
int state_size = sizeof(state_struct_t);

class ClockNamesAccessor: public dbm::ClockAccessor {
public:
    virtual const std::string& getClockName(cindex_t i) const {
        switch (i) {
			case 1: { static const std::string _n("time"); return _n; }
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

        }
        static const std::string _ref("0");
        return _ref;
    }
};
static ClockNamesAccessor clockNames = ClockNamesAccessor();

extern "C" int get_state_variable_count() 
{
    return 14;
}

extern "C" const char* get_state_variable_name(int var)
{
    switch (var)
    {
		case 0: return "L";
		case 1: return "Viking1";
		case 2: return "Viking2";
		case 3: return "Viking3";
		case 4: return "Viking4";
		case 5: return "Viking5";
		case 6: return "Viking6";
		case 7: return "Viking7";
		case 8: return "Viking8";
		case 9: return "Viking9";
		case 10: return "Viking10";
		case 11: return "Torch";
		case 12: return "fed";
		case 13: return "fed2";

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
    return 14;
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
    return 45;
}


static state_struct_t initial_state = {
	0, /* L */
	1, /* Viking1 */
	1, /* Viking2 */
	1, /* Viking3 */
	1, /* Viking4 */
	1, /* Viking5 */
	2, /* Viking6 */
	2, /* Viking7 */
	3, /* Viking8 */
	1, /* Viking9 */
	3, /* Viking10 */
	1, /* Torch */
	NULL, /* fed part */
};

/* Max clock constants {{{ */
int max_constants_Viking1[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,5,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,5,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking2[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking3[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking4[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking5[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking6[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking7[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking8[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking9[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Viking10[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,10,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_Torch[4][12] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
inline void extrapolate_max_constants (state_struct_t *out) {
	int local_max_consts[12] = {0, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, };
	int i;
	for (i = 0; i < 12; i++) {
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking1[out->Viking1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking2[out->Viking2][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking3[out->Viking3][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking4[out->Viking4][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking5[out->Viking5][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking6[out->Viking6][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking7[out->Viking7][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking8[out->Viking8][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking9[out->Viking9][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Viking10[out->Viking10][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_Torch[out->Torch][i] );
	}

    out->fed->extrapolateMaxBounds(local_max_consts);
}
/* }}} */

/* Invariants function {{{ */
inline bool apply_invariants (state_struct_t *out) {
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
	_s.fed = new dbm::fed_t(12);
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
	/* Transitions for Viking1 {{{*/
	switch( in->Viking1 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -5, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -5, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 2;
							out->Torch = 2;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -5, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -5, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 0;
							out->Torch = 2;
							out->fed->updateValue(2, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking1 = 0;
							out->Torch = 0;
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
	/* Transitions for Viking2 {{{*/
	switch( in->Viking2 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 2;
							out->Torch = 2;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 0;
							out->Torch = 2;
							out->fed->updateValue(3, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking2 = 0;
							out->Torch = 0;
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
	/* Transitions for Viking3 {{{*/
	switch( in->Viking3 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 2;
							out->Torch = 2;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 0;
							out->Torch = 2;
							out->fed->updateValue(4, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking3 = 0;
							out->Torch = 0;
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
	/* Transitions for Viking4 {{{*/
	switch( in->Viking4 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 2;
							out->Torch = 2;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 0;
							out->Torch = 2;
							out->fed->updateValue(5, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking4 = 0;
							out->Torch = 0;
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
	/* Transitions for Viking5 {{{*/
	switch( in->Viking5 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 2;
							out->Torch = 2;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 0;
							out->Torch = 2;
							out->fed->updateValue(6, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking5 = 0;
							out->Torch = 0;
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
	/* Transitions for Viking6 {{{*/
	switch( in->Viking6 ) {
		case 0: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 3;
							out->Torch = 2;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 3;
							out->Torch = 0;
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
		case 1:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 0;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 0;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 2: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 1;
							out->Torch = 2;
							out->fed->updateValue(7, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 1;
							out->Torch = 0;
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
		case 3:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 2;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 7, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking6 = 2;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
	/* Transitions for Viking7 {{{*/
	switch( in->Viking7 ) {
		case 0: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 3;
							out->Torch = 2;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 3;
							out->Torch = 0;
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
		case 1:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 0;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 0;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 2: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 1;
							out->Torch = 2;
							out->fed->updateValue(8, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 1;
							out->Torch = 0;
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
		case 3:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 2;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 8, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking7 = 2;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
	/* Transitions for Viking8 {{{*/
	switch( in->Viking8 ) {
		case 0: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 2;
							out->Torch = 2;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 2;
							out->Torch = 0;
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
		case 1:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 0;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 0;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 9, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 1;
							out->Torch = 2;
							out->fed->updateValue(9, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking8 = 1;
							out->Torch = 0;
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
	}
	/* }}} */
	/* Transitions for Viking9 {{{*/
	switch( in->Viking9 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 2;
							out->Torch = 2;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 10, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 0;
							out->Torch = 2;
							out->fed->updateValue(10, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking9 = 0;
							out->Torch = 0;
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
	/* Transitions for Viking10 {{{*/
	switch( in->Viking10 ) {
		case 0:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 1;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 1;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 1: /* safe */
			if( (in->L == 1) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 2;
							out->Torch = 2;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 2;
							out->Torch = 0;
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
		case 2:
			if( true ) {
				/* Syncs with Torch on release {{{*/
				switch( in->Torch ) {
					case 0: /* two */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 3;
							out->Torch = 3;
							/* nop */
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* one */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 11, -10, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 3;
							out->Torch = 1;
							/* nop */
							out->L = ((1 - out->L)%2);

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
		case 3: /* unsafe */
			if( (in->L == 0) ) {
				/* Syncs with Torch on take {{{*/
				switch( in->Torch ) {
					case 1: /* free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 0;
							out->Torch = 2;
							out->fed->updateValue(11, 0);
							/* nop */

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 2:
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							 ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->Viking10 = 0;
							out->Torch = 0;
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
	/* Transitions for Torch {{{*/
	switch( in->Torch ) {
		case 0: /* two */
		break;
		case 1: /* free */
		break;
		case 2:
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				/* nop */
				if (!(out->fed->isEmpty())) {
				out->Torch = 3;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 3: /* one */
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

int transition_dependency[14] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1, };

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
