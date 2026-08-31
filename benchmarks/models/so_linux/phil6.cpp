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
	uint32_t F0;
	uint32_t F1;
	uint32_t F2;
	uint32_t F3;
	uint32_t F4;
	uint32_t F5;
	uint32_t P0;
	uint32_t P1;
	uint32_t P2;
	uint32_t P3;
	uint32_t P4;
	uint32_t P5;
	dbm::fed_t *fed;
}
__attribute__((__packed__));
int state_size = sizeof(state_struct_t);

class ClockNamesAccessor: public dbm::ClockAccessor {
public:
    virtual const std::string& getClockName(cindex_t i) const {
        switch (i) {
			case 1: { static const std::string _n("c0"); return _n; }
			case 2: { static const std::string _n("c1"); return _n; }
			case 3: { static const std::string _n("c2"); return _n; }
			case 4: { static const std::string _n("c3"); return _n; }
			case 5: { static const std::string _n("c4"); return _n; }
			case 6: { static const std::string _n("c5"); return _n; }

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
		case 0: return "F0";
		case 1: return "F1";
		case 2: return "F2";
		case 3: return "F3";
		case 4: return "F4";
		case 5: return "F5";
		case 6: return "P0";
		case 7: return "P1";
		case 8: return "P2";
		case 9: return "P3";
		case 10: return "P4";
		case 11: return "P5";
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
    return 36;
}


static state_struct_t initial_state = {
	1, /* F0 */
	1, /* F1 */
	0, /* F2 */
	0, /* F3 */
	0, /* F4 */
	1, /* F5 */
	0, /* P0 */
	2, /* P1 */
	3, /* P2 */
	0, /* P3 */
	2, /* P4 */
	0, /* P5 */
	NULL, /* fed part */
};

/* Max clock constants {{{ */
int max_constants_F0[2][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_F1[2][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_F2[2][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_F3[2][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_F4[2][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_F5[2][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_P0[4][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_P1[4][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_P2[4][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_P3[4][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_P4[4][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,-dbm_INFINITY,},
};
int max_constants_P5[4][7] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,2,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
inline void extrapolate_max_constants (state_struct_t *out) {
	int local_max_consts[7] = {0, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, };
	int i;
	for (i = 0; i < 7; i++) {
		local_max_consts[i] = max( local_max_consts[i], max_constants_F0[out->F0][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_F1[out->F1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_F2[out->F2][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_F3[out->F3][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_F4[out->F4][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_F5[out->F5][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_P0[out->P0][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_P1[out->P1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_P2[out->P2][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_P3[out->P3][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_P4[out->P4][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_P5[out->P5][i] );
	}

    out->fed->extrapolateMaxBounds(local_max_consts);
}
/* }}} */

/* Invariants function {{{ */
inline bool apply_invariants (state_struct_t *out) {
	/* Invariants for P0 {{{*/
	switch( out->P0 ) {
		case 0: /* Think */
		break;
		case 1: /* Dropped */
		break;
		case 2: /* HoldL */
		break;
		case 3: /* Eat */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, 2, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for P1 {{{*/
	switch( out->P1 ) {
		case 0: /* Eat */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, 2, false); 
			if (!(true)) return false;
		break;
		case 1: /* Dropped */
		break;
		case 2: /* Think */
		break;
		case 3: /* HoldL */
		break;
	}
	/* Invariants for P2 {{{*/
	switch( out->P2 ) {
		case 0: /* Dropped */
		break;
		case 1: /* Eat */
			/* nop */ ; *(out->fed) &= dbm_constraint2(3, 0, 2, false); 
			if (!(true)) return false;
		break;
		case 2: /* HoldL */
		break;
		case 3: /* Think */
		break;
	}
	/* Invariants for P3 {{{*/
	switch( out->P3 ) {
		case 0: /* Think */
		break;
		case 1: /* Dropped */
		break;
		case 2: /* Eat */
			/* nop */ ; *(out->fed) &= dbm_constraint2(4, 0, 2, false); 
			if (!(true)) return false;
		break;
		case 3: /* HoldL */
		break;
	}
	/* Invariants for P4 {{{*/
	switch( out->P4 ) {
		case 0: /* HoldL */
		break;
		case 1: /* Dropped */
		break;
		case 2: /* Think */
		break;
		case 3: /* Eat */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, 2, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for P5 {{{*/
	switch( out->P5 ) {
		case 0: /* Think */
		break;
		case 1: /* HoldL */
		break;
		case 2: /* Eat */
			/* nop */ ; *(out->fed) &= dbm_constraint2(6, 0, 2, false); 
			if (!(true)) return false;
		break;
		case 3: /* Dropped */
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
	_s.fed = new dbm::fed_t(7);
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
	/* Transitions for F0 {{{*/
	switch( in->F0 ) {
		case 0: /* Used */
		break;
		case 1: /* Free */
		break;
	}
	/* }}} */
	/* Transitions for F1 {{{*/
	switch( in->F1 ) {
		case 0: /* Used */
		break;
		case 1: /* Free */
		break;
	}
	/* }}} */
	/* Transitions for F2 {{{*/
	switch( in->F2 ) {
		case 0: /* Free */
		break;
		case 1: /* Used */
		break;
	}
	/* }}} */
	/* Transitions for F3 {{{*/
	switch( in->F3 ) {
		case 0: /* Free */
		break;
		case 1: /* Used */
		break;
	}
	/* }}} */
	/* Transitions for F4 {{{*/
	switch( in->F4 ) {
		case 0: /* Free */
		break;
		case 1: /* Used */
		break;
	}
	/* }}} */
	/* Transitions for F5 {{{*/
	switch( in->F5 ) {
		case 0: /* Used */
		break;
		case 1: /* Free */
		break;
	}
	/* }}} */
	/* Transitions for P0 {{{*/
	switch( in->P0 ) {
		case 0: /* Think */
			if( true ) {
				/* Syncs with F0 on take0 {{{*/
				switch( in->F0 ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P0 = 2;
							out->F0 = 0;
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
		case 1: /* Dropped */
			if( true ) {
				/* Syncs with F1 on put1 {{{*/
				switch( in->F1 ) {
					case 0: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P0 = 0;
							out->F1 = 1;
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
		case 2: /* HoldL */
			if( true ) {
				/* Syncs with F1 on take1 {{{*/
				switch( in->F1 ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P0 = 3;
							out->F1 = 0;
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
		case 3: /* Eat */
			if( true ) {
				/* Syncs with F0 on put0 {{{*/
				switch( in->F0 ) {
					case 0: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 1, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P0 = 1;
							out->F0 = 1;
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
	/* Transitions for P1 {{{*/
	switch( in->P1 ) {
		case 0: /* Eat */
			if( true ) {
				/* Syncs with F1 on put1 {{{*/
				switch( in->F1 ) {
					case 0: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P1 = 1;
							out->F1 = 1;
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
		case 1: /* Dropped */
			if( true ) {
				/* Syncs with F2 on put2 {{{*/
				switch( in->F2 ) {
					case 1: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P1 = 2;
							out->F2 = 0;
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
		case 2: /* Think */
			if( true ) {
				/* Syncs with F1 on take1 {{{*/
				switch( in->F1 ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P1 = 3;
							out->F1 = 0;
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
		case 3: /* HoldL */
			if( true ) {
				/* Syncs with F2 on take2 {{{*/
				switch( in->F2 ) {
					case 0: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P1 = 0;
							out->F2 = 1;
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
	/* Transitions for P2 {{{*/
	switch( in->P2 ) {
		case 0: /* Dropped */
			if( true ) {
				/* Syncs with F3 on put3 {{{*/
				switch( in->F3 ) {
					case 1: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P2 = 3;
							out->F3 = 0;
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
		case 1: /* Eat */
			if( true ) {
				/* Syncs with F2 on put2 {{{*/
				switch( in->F2 ) {
					case 1: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 3, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P2 = 0;
							out->F2 = 0;
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
		case 2: /* HoldL */
			if( true ) {
				/* Syncs with F3 on take3 {{{*/
				switch( in->F3 ) {
					case 0: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P2 = 1;
							out->F3 = 1;
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
		case 3: /* Think */
			if( true ) {
				/* Syncs with F2 on take2 {{{*/
				switch( in->F2 ) {
					case 0: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P2 = 2;
							out->F2 = 1;
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
	/* Transitions for P3 {{{*/
	switch( in->P3 ) {
		case 0: /* Think */
			if( true ) {
				/* Syncs with F3 on take3 {{{*/
				switch( in->F3 ) {
					case 0: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P3 = 3;
							out->F3 = 1;
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
		case 1: /* Dropped */
			if( true ) {
				/* Syncs with F4 on put4 {{{*/
				switch( in->F4 ) {
					case 1: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P3 = 0;
							out->F4 = 0;
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
		case 2: /* Eat */
			if( true ) {
				/* Syncs with F3 on put3 {{{*/
				switch( in->F3 ) {
					case 1: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 4, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P3 = 1;
							out->F3 = 0;
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
		case 3: /* HoldL */
			if( true ) {
				/* Syncs with F4 on take4 {{{*/
				switch( in->F4 ) {
					case 0: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P3 = 2;
							out->F4 = 1;
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
	/* Transitions for P4 {{{*/
	switch( in->P4 ) {
		case 0: /* HoldL */
			if( true ) {
				/* Syncs with F5 on take5 {{{*/
				switch( in->F5 ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P4 = 3;
							out->F5 = 0;
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
		case 1: /* Dropped */
			if( true ) {
				/* Syncs with F5 on put5 {{{*/
				switch( in->F5 ) {
					case 0: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P4 = 2;
							out->F5 = 1;
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
		case 2: /* Think */
			if( true ) {
				/* Syncs with F4 on take4 {{{*/
				switch( in->F4 ) {
					case 0: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P4 = 0;
							out->F4 = 1;
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
		case 3: /* Eat */
			if( true ) {
				/* Syncs with F4 on put4 {{{*/
				switch( in->F4 ) {
					case 1: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P4 = 1;
							out->F4 = 0;
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
	/* Transitions for P5 {{{*/
	switch( in->P5 ) {
		case 0: /* Think */
			if( true ) {
				/* Syncs with F5 on take5 {{{*/
				switch( in->F5 ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P5 = 1;
							out->F5 = 0;
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
		case 1: /* HoldL */
			if( true ) {
				/* Syncs with F0 on take0 {{{*/
				switch( in->F0 ) {
					case 1: /* Free */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P5 = 2;
							out->F0 = 0;
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
		case 2: /* Eat */
			if( true ) {
				/* Syncs with F5 on put5 {{{*/
				switch( in->F5 ) {
					case 0: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 6, -1, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P5 = 3;
							out->F5 = 1;
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
		case 3: /* Dropped */
			if( true ) {
				/* Syncs with F0 on put0 {{{*/
				switch( in->F0 ) {
					case 0: /* Used */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->P5 = 0;
							out->F0 = 1;
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
