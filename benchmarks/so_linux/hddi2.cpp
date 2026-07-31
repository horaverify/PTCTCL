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

const uint32_t SA = 20;
const uint32_t td = 0;
const uint32_t TRTT = 120;

struct state_struct_t
{
	uint32_t RING;
	uint32_t ST1;
	uint32_t ST2;
	dbm::fed_t *fed;
}
__attribute__((__packed__));
int state_size = sizeof(state_struct_t);

class ClockNamesAccessor: public dbm::ClockAccessor {
public:
    virtual const std::string& getClockName(cindex_t i) const {
        switch (i) {
			case 1: { static const std::string _n("RING_x"); return _n; }
			case 2: { static const std::string _n("ST1_x"); return _n; }
			case 3: { static const std::string _n("ST1_y"); return _n; }
			case 4: { static const std::string _n("ST1_z"); return _n; }
			case 5: { static const std::string _n("ST2_x"); return _n; }
			case 6: { static const std::string _n("ST2_y"); return _n; }
			case 7: { static const std::string _n("ST2_z"); return _n; }

        }
        static const std::string _ref("0");
        return _ref;
    }
};
static ClockNamesAccessor clockNames = ClockNamesAccessor();

extern "C" int get_state_variable_count() 
{
    return 5;
}

extern "C" const char* get_state_variable_name(int var)
{
    switch (var)
    {
		case 0: return "RING";
		case 1: return "ST1";
		case 2: return "ST2";
		case 3: return "fed";
		case 4: return "fed2";

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
    return 5;
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
    return 20;
}


static state_struct_t initial_state = {
	2, /* RING */
	4, /* ST1 */
	0, /* ST2 */
	NULL, /* fed part */
};

/* Max clock constants {{{ */
int max_constants_RING[4][8] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_ST1[6][8] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};
int max_constants_ST2[6][8] = {
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
	{-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,-dbm_INFINITY,},
};

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
inline void extrapolate_max_constants (state_struct_t *out) {
	int local_max_consts[8] = {0, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, -dbm_INFINITY, };
	int i;
	for (i = 0; i < 8; i++) {
		local_max_consts[i] = max( local_max_consts[i], max_constants_RING[out->RING][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_ST1[out->ST1][i] );
		local_max_consts[i] = max( local_max_consts[i], max_constants_ST2[out->ST2][i] );
	}

    out->fed->extrapolateMaxBounds(local_max_consts);
}
/* }}} */

/* Invariants function {{{ */
inline bool apply_invariants (state_struct_t *out) {
	/* Invariants for RING {{{*/
	switch( out->RING ) {
		case 0: /* ring_to_2 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, td, false); 
			if (!(true)) return false;
		break;
		case 1: /* ring_2 */
			/* nop */ ; 
		break;
		case 2: /* ring_to_1 */
			/* nop */ ; *(out->fed) &= dbm_constraint2(1, 0, td, false); 
			if (!(true)) return false;
		break;
		case 3: /* ring_1 */
			/* nop */ ; 
		break;
	}
	/* Invariants for ST1 {{{*/
	switch( out->ST1 ) {
		case 0: /* station_y_sync */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, SA, false); 
			if (!(true)) return false;
		break;
		case 1: /* station_y_async */
			/* nop */ ; *(out->fed) &= dbm_constraint2(3, 0, TRTT, false); 
			if (!(true)) return false;
		break;
		case 2: /* station_z_async */
			/* nop */ ; *(out->fed) &= dbm_constraint2(4, 0, TRTT, false); 
			if (!(true)) return false;
		break;
		case 3: /* station_y_idle */
			/* nop */ ; 
		break;
		case 4: /* station_z_idle */
			/* nop */ ; 
		break;
		case 5: /* station_z_sync */
			/* nop */ ; *(out->fed) &= dbm_constraint2(2, 0, SA, false); 
			if (!(true)) return false;
		break;
	}
	/* Invariants for ST2 {{{*/
	switch( out->ST2 ) {
		case 0: /* station_z_idle */
			/* nop */ ; 
		break;
		case 1: /* station_z_sync */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, SA, false); 
			if (!(true)) return false;
		break;
		case 2: /* station_z_async */
			/* nop */ ; *(out->fed) &= dbm_constraint2(7, 0, TRTT, false); 
			if (!(true)) return false;
		break;
		case 3: /* station_y_idle */
			/* nop */ ; 
		break;
		case 4: /* station_y_sync */
			/* nop */ ; *(out->fed) &= dbm_constraint2(5, 0, SA, false); 
			if (!(true)) return false;
		break;
		case 5: /* station_y_async */
			/* nop */ ; *(out->fed) &= dbm_constraint2(6, 0, TRTT, false); 
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
	_s.fed = new dbm::fed_t(8);
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
	/* Transitions for RING {{{*/
	switch( in->RING ) {
		case 0: /* ring_to_2 */
			if( true ) {
				/* Syncs with ST2 on tt2 {{{*/
				switch( in->ST2 ) {
					case 0: /* station_z_idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(1, 0, td, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->RING = 1;
							out->ST2 = 1;
							/* nop */
							out->fed->updateValue(6, 0);out->fed->updateValue(5, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 3: /* station_y_idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(1, 0, td, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->RING = 1;
							out->ST2 = 4;
							/* nop */
							out->fed->updateValue(7, 0);out->fed->updateValue(5, 0);

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
		case 1: /* ring_2 */
		break;
		case 2: /* ring_to_1 */
			if( true ) {
				/* Syncs with ST1 on tt1 {{{*/
				switch( in->ST1 ) {
					case 3: /* station_y_idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(1, 0, td, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->RING = 3;
							out->ST1 = 0;
							/* nop */
							out->fed->updateValue(4, 0);out->fed->updateValue(2, 0);

                        system_in_deadlock = false;
                        transition_info.group = -1;
                        do_callback(out, &transition_info, callback, arg);
                        ++states_emitted;
							}
						}
					break;
					case 4: /* station_z_idle */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(1, 0, td, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->RING = 3;
							out->ST1 = 5;
							/* nop */
							out->fed->updateValue(3, 0);out->fed->updateValue(2, 0);

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
		case 3: /* ring_1 */
		break;
	}
	/* }}} */
	/* Transitions for ST1 {{{*/
	switch( in->ST1 ) {
		case 0: /* station_y_sync */
			if( true ) {
				/* Syncs with RING on rt1 {{{*/
				switch( in->RING ) {
					case 3: /* ring_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -SA, false); *(out->fed) &= dbm_constraint2(0, 3, -TRTT, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST1 = 4;
							out->RING = 0;
							/* nop */
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
			}
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 2, -SA, false); *(out->fed) &= dbm_constraint2(3, 0, TRTT, true); 
				if (!(out->fed->isEmpty())) {
				out->ST1 = 1;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 1: /* station_y_async */
			if( true ) {
				/* Syncs with RING on rt1 {{{*/
				switch( in->RING ) {
					case 3: /* ring_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST1 = 4;
							out->RING = 0;
							/* nop */
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
			}
		break;
		case 2: /* station_z_async */
			if( true ) {
				/* Syncs with RING on rt1 {{{*/
				switch( in->RING ) {
					case 3: /* ring_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST1 = 3;
							out->RING = 0;
							/* nop */
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
			}
		break;
		case 3: /* station_y_idle */
		break;
		case 4: /* station_z_idle */
		break;
		case 5: /* station_z_sync */
			if( true ) {
				/* Syncs with RING on rt1 {{{*/
				switch( in->RING ) {
					case 3: /* ring_1 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 2, -SA, false); *(out->fed) &= dbm_constraint2(0, 4, -TRTT, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST1 = 3;
							out->RING = 0;
							/* nop */
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
			}
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 2, -SA, false); *(out->fed) &= dbm_constraint2(4, 0, TRTT, true); 
				if (!(out->fed->isEmpty())) {
				out->ST1 = 2;
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
	/* Transitions for ST2 {{{*/
	switch( in->ST2 ) {
		case 0: /* station_z_idle */
		break;
		case 1: /* station_z_sync */
			if( true ) {
				/* Syncs with RING on rt2 {{{*/
				switch( in->RING ) {
					case 1: /* ring_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -SA, false); *(out->fed) &= dbm_constraint2(0, 7, -TRTT, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST2 = 3;
							out->RING = 2;
							/* nop */
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
			}
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 5, -SA, false); *(out->fed) &= dbm_constraint2(7, 0, TRTT, true); 
				if (!(out->fed->isEmpty())) {
				out->ST2 = 2;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 2: /* station_z_async */
			if( true ) {
				/* Syncs with RING on rt2 {{{*/
				switch( in->RING ) {
					case 1: /* ring_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST2 = 3;
							out->RING = 2;
							/* nop */
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
			}
		break;
		case 3: /* station_y_idle */
		break;
		case 4: /* station_y_sync */
			if( true ) {
				/* Syncs with RING on rt2 {{{*/
				switch( in->RING ) {
					case 1: /* ring_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							*(out->fed) &= dbm_constraint2(0, 5, -SA, false); *(out->fed) &= dbm_constraint2(0, 6, -TRTT, false);  ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST2 = 0;
							out->RING = 2;
							/* nop */
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
			}
			if( true ) {
				*out = *in;
				out->fed = new dbm::fed_t(*(in->fed));
				*(out->fed) &= dbm_constraint2(0, 5, -SA, false); *(out->fed) &= dbm_constraint2(6, 0, TRTT, true); 
				if (!(out->fed->isEmpty())) {
				out->ST2 = 5;
				/* nop */

				system_in_deadlock = false;
				transition_info.group = -1;
				do_callback(out, &transition_info, callback, arg);
				++states_emitted;
				}

			}
		break;
		case 5: /* station_y_async */
			if( true ) {
				/* Syncs with RING on rt2 {{{*/
				switch( in->RING ) {
					case 1: /* ring_2 */
						if( true ) {
							*out = *in;
							out->fed = new dbm::fed_t(*(in->fed));
							/* nop */ ; /* nop */
							if (!(out->fed->isEmpty())) {
							out->ST2 = 0;
							out->RING = 2;
							/* nop */
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

int transition_dependency[5] = { 1,1,1,1,1, };

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
