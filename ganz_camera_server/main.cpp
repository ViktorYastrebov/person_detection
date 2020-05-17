#include <signal.h> 
#include <winsock2.h>
#include <ws2tcpip.h>
#include <memory>
#include <cstdint>
#include <iostream>
#include <evhttp.h>

char uri_root[512];

void dump_request_cb(struct evhttp_request *req, void *arg)
{
    const char *cmdtype;
    struct evkeyvalq *headers;
    struct evkeyval *header;
    struct evbuffer *buf;

    switch (evhttp_request_get_command(req)) {
    case EVHTTP_REQ_GET: cmdtype = "GET"; break;
    case EVHTTP_REQ_POST: cmdtype = "POST"; break;
    case EVHTTP_REQ_HEAD: cmdtype = "HEAD"; break;
    case EVHTTP_REQ_PUT: cmdtype = "PUT"; break;
    case EVHTTP_REQ_DELETE: cmdtype = "DELETE"; break;
    case EVHTTP_REQ_OPTIONS: cmdtype = "OPTIONS"; break;
    case EVHTTP_REQ_TRACE: cmdtype = "TRACE"; break;
    case EVHTTP_REQ_CONNECT: cmdtype = "CONNECT"; break;
    case EVHTTP_REQ_PATCH: cmdtype = "PATCH"; break;
    default: cmdtype = "unknown"; break;
    }

    printf("Received a %s request for %s\nHeaders:\n",
        cmdtype, evhttp_request_get_uri(req));

    headers = evhttp_request_get_input_headers(req);
    for (header = headers->tqh_first; header;
        header = header->next.tqe_next) {
        printf("  %s: %s\n", header->key, header->value);
    }

    buf = evhttp_request_get_input_buffer(req);
    puts("Input data: <<<");
    while (evbuffer_get_length(buf)) {
        int n;
        char cbuf[128];
        n = evbuffer_remove(buf, cbuf, sizeof(cbuf));
        if (n > 0)
            (void)fwrite(cbuf, 1, n, stdout);
    }
    puts(">>>");

    evhttp_send_reply(req, 200, "OK", NULL);
}

int display_listen_sock(struct evhttp_bound_socket *handle)
{
    struct sockaddr_storage ss;
    evutil_socket_t fd;
    ev_socklen_t socklen = sizeof(ss);
    char addrbuf[128];
    void *inaddr;
    const char *addr;
    int got_port = -1;

    fd = evhttp_bound_socket_get_fd(handle);
    memset(&ss, 0, sizeof(ss));
    if (getsockname(fd, (struct sockaddr *)&ss, &socklen)) {
        std::cout << "getsockname() failed" << std::endl;
        return 1;
    }

    if (ss.ss_family == AF_INET) {
        got_port = ntohs(((struct sockaddr_in*)&ss)->sin_port);
        inaddr = &((struct sockaddr_in*)&ss)->sin_addr;
    }
    else if (ss.ss_family == AF_INET6) {
        got_port = ntohs(((struct sockaddr_in6*)&ss)->sin6_port);
        inaddr = &((struct sockaddr_in6*)&ss)->sin6_addr;
    }
//#ifdef EVENT__HAVE_STRUCT_SOCKADDR_UN
//    else if (ss.ss_family == AF_UNIX) {
//        std::cout << "Listening on " << ((struct sockaddr_un*)&ss)->sun_path << std::endl;
//        return 0;
//    }
//#endif
    else {
        std::cout << "Weird address family " << ss.ss_family << std::endl;
        return 1;
    }

    addr = evutil_inet_ntop(ss.ss_family, inaddr, addrbuf,
        sizeof(addrbuf));
    if (addr) {
        std::cout << "Listening on " << addr << ":" << got_port << std::endl;
        //printf("Listening on %s:%d\n", addr, got_port);
        //evutil_snprintf(uri_root, sizeof(uri_root),
        //    "http://%s:%d", addr, got_port);
        //std::cout << "http://" << uri_root << ":" << got_port << std::endl;
    }
    else {
        //fprintf(stderr, "evutil_inet_ntop failed\n");
        std::cout << "evutil_inet_ntop failed" << std::endl;
        return 1;
    }

    return 0;
}

void do_term(evutil_socket_t sig, short events, void *arg)
{
    struct event_base *base = static_cast<event_base*>(arg);
    event_base_loopbreak(base);
    std::cout << "Got " << sig << ", Terminating" << std::endl;
}

int main()
{
    const char *host = "127.0.0.1";
    uint16_t port = 5555;

    struct event_config *cfg = nullptr;
    struct event_base *base = nullptr;
    struct evhttp *http = nullptr;
    struct evhttp_bound_socket *handle = nullptr;
    struct evconnlistener *lev = nullptr;
    struct event *term = nullptr;

    WORD wVersionRequested;
    WSADATA wsaData;
    wVersionRequested = MAKEWORD(2, 2);
    WSAStartup(wVersionRequested, &wsaData);

    {
        cfg = event_config_new();
        base = event_base_new_with_config(cfg);
        if (!base) {
            std::cout << "Couldn't create an event_base: exiting" << std::endl;
            WSACleanup();
            return 1;
        }
        event_config_free(cfg);
        cfg = nullptr;

        /* Create a new evhttp object to handle requests. */
        http = evhttp_new(base);
        if (!http) {
            std::cout << "couldn't create evhttp. Exiting" << std::endl;
            return 1;
        }

        evhttp_set_cb(http, "", dump_request_cb, nullptr);

        // We want to accept arbitrary requests, so we need to set a "generic" cb.
        // We can also add callbacks for specific paths.
        //void(*OnReq)(evhttp_request *req, void *) = [](evhttp_request *req, void *)
        //{
        //    auto *OutBuf = evhttp_request_get_output_buffer(req);
        //    if (!OutBuf)
        //        return;
        //    evbuffer_add_printf(OutBuf, "<html><body><center><h1>Hello World!</h1></center></body></html>");
        //    evhttp_send_reply(req, HTTP_OK, "", OutBuf);
        //};
        //evhttp_set_gencb(http, send_document_cb, argv[1]);
        evhttp_set_gencb(http, dump_request_cb, nullptr);
        
        handle = evhttp_bind_socket_with_handle(http, host, port);
        if (!handle) {
            std::cout << "couldn't bind to port " << port << ", Exiting" << std::endl;
            if (cfg) {
                event_config_free(cfg);
            }
            if (http) {
                evhttp_free(http);
            }
            if (term) {
                event_free(term);
            }
            if (base) {
                event_base_free(base);
            }
            WSACleanup();
            return 1;
        }

        if (display_listen_sock(handle)) {
            if (cfg) {
                event_config_free(cfg);
            }
            if (http) {
                evhttp_free(http);
            }
            if (term) {
                event_free(term);
            }
            if (base) {
                event_base_free(base);
            }

            WSACleanup();
            return 1;
        }
        // typedef void (*event_callback_fn)(evutil_socket_t, short, void *);
        //term = evsignal_new(base, SIGINT, do_term, base);
        term = event_new(base, SIGINT, EV_SIGNAL | EV_PERSIST, do_term, base);
        if (!term) {
            if (cfg) {
                event_config_free(cfg);
            }
            if (http) {
                evhttp_free(http);
            }
            if (term) {
                event_free(term);
            }
            if (base) {
                event_base_free(base);
            }
            WSACleanup();
            return 1;
        }
        if (event_add(term, NULL)) {
            if (cfg) {
                event_config_free(cfg);
            }
            if (http) {
                evhttp_free(http);
            }
            if (term) {
                event_free(term);
            }
            if (base) {
                event_base_free(base);
            }
            WSACleanup();
            return 1;
        }
        event_base_dispatch(base);
    }

    if (cfg) {
        event_config_free(cfg);
    }
    if (http) {
        evhttp_free(http);
    }
    if (term) {
        event_free(term);
    }
    if (base) {
        event_base_free(base);
    }

    WSACleanup();
    return 0;
}