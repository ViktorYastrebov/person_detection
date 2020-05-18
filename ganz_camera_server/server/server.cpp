#include "server.h"


HttpServer *global_ptr = 0;

void terminate_wrapper(evutil_socket_t sig, short events, void *arg)
{
    if (global_ptr) {
        global_ptr->default_signal_handler(sig, events, arg);
    }
}

void request_wrapper(evhttp_request *req, void *arg) {
    if (global_ptr) {
        global_ptr->default_handler(req, arg);
    }
}


HttpServer::WSAContext::WSAContext(int major_verion, int minor_version) {
    WORD wVersionRequested = MAKEWORD(major_verion, minor_version);
    WSAStartup(wVersionRequested, &wsaData_);
}

HttpServer::WSAContext::~WSAContext() {
    WSACleanup();
}

//HttpServer::EventBaseContext::EventBaseContext()
//    :base(nullptr)
//{}
//
//HttpServer::EventBaseContext::EventBaseContext(event_config *config)
//    :base(event_base_new_with_config(config))
//{
//    if (!base) {
//        throw std::exception("Couldn't create an event_base: exiting");
//    }
//}
//
//HttpServer::EventBaseContext::~EventBaseContext() {
//    if (base) {
//        event_base_free(base);
//    }
//}

//HttpServer::EventHTTPContext::EventHTTPContext()
//    : http(nullptr)
//{}
//
//HttpServer::EventHTTPContext::EventHTTPContext(EventBaseContext &base)
//    :http(evhttp_new(base.base))
//{
//    if (!http) {
//        throw std::exception("couldn't create evhttp. Exiting");
//    }
//}
//
//HttpServer::EventHTTPContext::~EventHTTPContext() {
//    if (http) {
//        evhttp_free(http);
//    }
//    http = nullptr;
//}
//HttpServer::EventHTTPContext::EventHTTPContext(EventHTTPContext &&cpy) {
//    http = std::move(cpy.http);
//    cpy.http = nullptr;
//}
//
//HttpServer::EventHTTPContext & HttpServer::EventHTTPContext::operator = (EventHTTPContext &&cpy) {
//    if (http) {
//        evhttp_free(http);
//    }
//    http = std::move(cpy.http);
//    cpy.http = nullptr;
//    return *this;
//}

HttpServer::HttpServer(const std::string &host, std::uint16_t port, std::ostream & output)
    : host_(host)
    , port_(port)
    , output_(output)
    , wsa_context_(2,2)
{
    struct event_config *cfg = event_config_new();
    //base_ = EventBaseContext(cfg);
    base_ = event_base_new_with_config(cfg);
    if (!base_) {
        throw std::exception("Couldn't create an event_base: exiting");
    }

    event_config_free(cfg);
    //http_ = EventHTTPContext(base_);
    http_ = evhttp_new(base_);
    if (!http_) {
        throw std::exception("couldn't create evhttp. Exiting");
    }

    evhttp_set_gencb(http_, request_wrapper, nullptr);

    handle_ = evhttp_bind_socket_with_handle(http_, host_.c_str(), port_);
    if (!handle_) {
        std::string error = "couldn't bind to port " + std::to_string(port_) + ", Exiting";
        throw std::exception(error.c_str());
    }

    if (!display_listen_socket()) {
        throw std::exception("Linsten failed");
    }

    termination_event_ = event_new(base_, SIGINT, EV_SIGNAL | EV_PERSIST,
                                    terminate_wrapper,
                                    base_);
    if (!termination_event_) {
        throw std::exception("Create terminate handle has failed");
    }
    if (event_add(termination_event_, nullptr)) {
        throw std::exception("Add terminate handle has failed");
    }
    global_ptr = this;
    event_base_dispatch(base_);
}

HttpServer::~HttpServer() 
{
    if (http_) {
        evhttp_free(http_);
        http_ = nullptr;
    }

    if (termination_event_) {
        event_free(termination_event_);
        termination_event_ = nullptr;
    }

    if (base_) {
        event_base_free(base_);
        base_ = nullptr;
    }

    global_ptr = nullptr;
}

void HttpServer::default_handler(struct evhttp_request *req, void *arg) {
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

bool HttpServer::display_listen_socket() {
    struct sockaddr_storage ss;
    evutil_socket_t fd;
    ev_socklen_t socklen = sizeof(ss);
    char addrbuf[128];
    void *inaddr;
    const char *addr;
    int got_port = -1;

    fd = evhttp_bound_socket_get_fd(handle_);
    memset(&ss, 0, sizeof(ss));
    if (getsockname(fd, (struct sockaddr *)&ss, &socklen)) {
        output_ << "getsockname() failed" << std::endl;
        return false;
    }

    if (ss.ss_family == AF_INET) {
        got_port = ntohs(((struct sockaddr_in*)&ss)->sin_port);
        inaddr = &((struct sockaddr_in*)&ss)->sin_addr;
    }
    else if (ss.ss_family == AF_INET6) {
        got_port = ntohs(((struct sockaddr_in6*)&ss)->sin6_port);
        inaddr = &((struct sockaddr_in6*)&ss)->sin6_addr;
    } else {
        output_ << "Weird address family " << ss.ss_family << std::endl;
        return false;
    }

    addr = evutil_inet_ntop(ss.ss_family, inaddr, addrbuf,
        sizeof(addrbuf));
    if (addr) {
        output_ << "Listening on " << addr << ":" << got_port << std::endl;
        url_ = "http://" + std::string(addr) + ":" + std::to_string(got_port);
    } else {
        output_ << "evutil_inet_ntop failed" << std::endl;
        return false;
    }
    return true;
}

void HttpServer::default_signal_handler(evutil_socket_t sig, short events, void *arg) {
    struct event_base *base = static_cast<event_base*>(arg);
    event_base_loopbreak(base);
    output_ << "Got " << sig << ", Terminating" << std::endl;
}

const std::string & HttpServer::getURL() const {
    return url_;
}
