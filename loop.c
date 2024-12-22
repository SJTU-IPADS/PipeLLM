#include <sys/resource.h>
#include <unistd.h>
int main() {
    int which = PRIO_PROCESS;
    id_t pid;
    int priority = 19;
    int ret;

    pid = getpid();
    ret = setpriority(which, pid, priority);
    while (1);
}